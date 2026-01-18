import operator
import os
from io import BytesIO
from ..lazy_import import lazy_import
import patiencediff
import gzip
from breezy import (
from breezy.bzr import (
from breezy.bzr import pack_repo
from breezy.i18n import gettext
from .. import annotate, errors, osutils
from .. import transport as _mod_transport
from ..bzr.versionedfile import (AbsentContentFactory, ConstantMapper,
from ..errors import InternalBzrError, InvalidRevisionId, RevisionNotPresent
from ..osutils import contains_whitespace, sha_string, sha_strings, split_lines
from ..transport import NoSuchFile
from . import index as _mod_index
def add_records(self, records, random_id=False, missing_compression_parents=False):
    """Add multiple records to the index.

        This function does not insert data into the Immutable GraphIndex
        backing the KnitGraphIndex, instead it prepares data for insertion by
        the caller and checks that it is safe to insert then calls
        self._add_callback with the prepared GraphIndex nodes.

        :param records: a list of tuples:
                         (key, options, access_memo, parents).
        :param random_id: If True the ids being added were randomly generated
            and no check for existence will be performed.
        :param missing_compression_parents: If True the records being added are
            only compressed against texts already in the index (or inside
            records). If False the records all refer to unavailable texts (or
            texts inside records) as compression parents.
        """
    if not self._add_callback:
        raise errors.ReadOnlyError(self)
    keys = {}
    compression_parents = set()
    key_dependencies = self._key_dependencies
    for key, options, access_memo, parents in records:
        if self._parents:
            parents = tuple(parents)
            if key_dependencies is not None:
                key_dependencies.add_references(key, parents)
        index, pos, size = access_memo
        if b'no-eol' in options:
            value = b'N'
        else:
            value = b' '
        value += b'%d %d' % (pos, size)
        if not self._deltas:
            if b'line-delta' in options:
                raise KnitCorrupt(self, 'attempt to add line-delta in non-delta knit')
        if self._parents:
            if self._deltas:
                if b'line-delta' in options:
                    node_refs = (parents, (parents[0],))
                    if missing_compression_parents:
                        compression_parents.add(parents[0])
                else:
                    node_refs = (parents, ())
            else:
                node_refs = (parents,)
        else:
            if parents:
                raise KnitCorrupt(self, 'attempt to add node with parents in parentless index.')
            node_refs = ()
        keys[key] = (value, node_refs)
    if not random_id:
        present_nodes = self._get_entries(keys)
        for index, key, value, node_refs in present_nodes:
            parents = node_refs[:1]
            passed = static_tuple.as_tuples(keys[key])
            passed_parents = passed[1][:1]
            if value[0:1] != keys[key][0][0:1] or parents != passed_parents:
                node_refs = static_tuple.as_tuples(node_refs)
                raise KnitCorrupt(self, 'inconsistent details in add_records: %s %s' % ((value, node_refs), passed))
            del keys[key]
    result = []
    if self._parents:
        for key, (value, node_refs) in keys.items():
            result.append((key, value, node_refs))
    else:
        for key, (value, node_refs) in keys.items():
            result.append((key, value))
    self._add_callback(result)
    if missing_compression_parents:
        compression_parents.difference_update(keys)
        self._missing_compression_parents.update(compression_parents)
    self._missing_compression_parents.difference_update(keys)