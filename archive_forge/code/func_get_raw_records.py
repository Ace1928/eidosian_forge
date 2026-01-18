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
def get_raw_records(self, memos_for_retrieval):
    """Get the raw bytes for a records.

        :param memos_for_retrieval: An iterable containing the access memo for
            retrieving the bytes.
        :return: An iterator over the bytes of the records.
        """
    request_lists = []
    current_prefix = None
    for key, offset, length in memos_for_retrieval:
        if current_prefix == key[:-1]:
            current_list.append((offset, length))
        else:
            if current_prefix is not None:
                request_lists.append((current_prefix, current_list))
            current_prefix = key[:-1]
            current_list = [(offset, length)]
    if current_prefix is not None:
        request_lists.append((current_prefix, current_list))
    for prefix, read_vector in request_lists:
        path = self._mapper.map(prefix) + '.knit'
        for pos, data in self._transport.readv(path, read_vector):
            yield data