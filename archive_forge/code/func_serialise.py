import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def serialise(self, store):
    """Serialise the node to store.

        :param store: A VersionedFiles honouring the CHK extensions.
        :return: An iterable of the keys inserted by this operation.
        """
    for node in self._items.values():
        if isinstance(node, StaticTuple):
            continue
        if node._key is not None:
            continue
        for key in node.serialise(store):
            yield key
    lines = [b'chknode:\n']
    lines.append(b'%d\n' % self._maximum_size)
    lines.append(b'%d\n' % self._key_width)
    lines.append(b'%d\n' % self._len)
    if self._search_prefix is None:
        raise AssertionError('_search_prefix should not be None')
    lines.append(b'%s\n' % (self._search_prefix,))
    prefix_len = len(self._search_prefix)
    for prefix, node in sorted(self._items.items()):
        if isinstance(node, StaticTuple):
            key = node[0]
        else:
            key = node._key[0]
        serialised = b'%s\x00%s\n' % (prefix, key)
        if not serialised.startswith(self._search_prefix):
            raise AssertionError('prefixes mismatch: %s must start with %s' % (serialised, self._search_prefix))
        lines.append(serialised[prefix_len:])
    sha1, _, _ = store.add_lines((None,), (), lines)
    self._key = StaticTuple(b'sha1:' + sha1).intern()
    _get_cache()[self._key] = b''.join(lines)
    yield self._key