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
def get_build_details(self, keys):
    """Get the method, index_memo and compression parent for version_ids.

        Ghosts are omitted from the result.

        :param keys: An iterable of keys.
        :return: A dict of key:
            (index_memo, compression_parent, parents, record_details).
            index_memo
                opaque structure to pass to read_records to extract the raw
                data
            compression_parent
                Content that this record is built upon, may be None
            parents
                Logical parents of this node
            record_details
                extra information about the content which needs to be passed to
                Factory.parse_record
        """
    self._check_read()
    result = {}
    entries = self._get_entries(keys, False)
    for entry in entries:
        key = entry[1]
        if not self._parents:
            parents = ()
        else:
            parents = entry[3][0]
        if not self._deltas:
            compression_parent_key = None
        else:
            compression_parent_key = self._compression_parent(entry)
        noeol = entry[2][0:1] == b'N'
        if compression_parent_key:
            method = 'line-delta'
        else:
            method = 'fulltext'
        result[key] = (self._node_to_position(entry), compression_parent_key, parents, (method, noeol))
    return result