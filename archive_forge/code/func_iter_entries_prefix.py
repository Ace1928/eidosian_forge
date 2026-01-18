from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
def iter_entries_prefix(self, keys):
    """Iterate over keys within the index using prefix matching.

        Prefix matching is applied within the tuple of a key, not to within
        the bytestring of each key element. e.g. if you have the keys ('foo',
        'bar'), ('foobar', 'gam') and do a prefix search for ('foo', None) then
        only the former key is returned.

        WARNING: Note that this method currently causes a full index parse
        unconditionally (which is reasonably appropriate as it is a means for
        thunking many small indices into one larger one and still supplies
        iter_all_entries at the thunk layer).

        :param keys: An iterable providing the key prefixes to be retrieved.
            Each key prefix takes the form of a tuple the length of a key, but
            with the last N elements 'None' rather than a regular bytestring.
            The first element cannot be 'None'.
        :return: An iterable as per iter_all_entries, but restricted to the
            keys with a matching prefix to those supplied. No additional keys
            will be returned, and every match that is in the index will be
            returned.
        """
    keys = sorted(set(keys))
    if not keys:
        return
    if self._key_count is None:
        self._get_root_node()
    nodes = {}
    if self.node_ref_lists:
        if self._key_length == 1:
            for _1, key, value, refs in self.iter_all_entries():
                nodes[key] = (value, refs)
        else:
            nodes_by_key = {}
            for _1, key, value, refs in self.iter_all_entries():
                key_value = (key, value, refs)
                key_dict = nodes_by_key
                for subkey in key[:-1]:
                    key_dict = key_dict.setdefault(subkey, {})
                key_dict[key[-1]] = key_value
    elif self._key_length == 1:
        for _1, key, value in self.iter_all_entries():
            nodes[key] = value
    else:
        nodes_by_key = {}
        for _1, key, value in self.iter_all_entries():
            key_value = (key, value)
            key_dict = nodes_by_key
            for subkey in key[:-1]:
                key_dict = key_dict.setdefault(subkey, {})
            key_dict[key[-1]] = key_value
    if self._key_length == 1:
        for key in keys:
            index._sanity_check_key(self, key)
            try:
                if self.node_ref_lists:
                    value, node_refs = nodes[key]
                    yield (self, key, value, node_refs)
                else:
                    yield (self, key, nodes[key])
            except KeyError:
                pass
        return
    yield from index._iter_entries_prefix(self, nodes_by_key, keys)