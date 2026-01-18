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
def knit_network_to_record(storage_kind, bytes, line_end):
    """Convert a network record to a record object.

    :param storage_kind: The storage kind of the record.
    :param bytes: The bytes of the record on the network.
    """
    start = line_end
    line_end = bytes.find(b'\n', start)
    key = tuple(bytes[start:line_end].split(b'\x00'))
    start = line_end + 1
    line_end = bytes.find(b'\n', start)
    parent_line = bytes[start:line_end]
    if parent_line == b'None:':
        parents = None
    else:
        parents = tuple([tuple(segment.split(b'\x00')) for segment in parent_line.split(b'\t') if segment])
    start = line_end + 1
    noeol = bytes[start:start + 1] == b'N'
    if 'ft' in storage_kind:
        method = 'fulltext'
    else:
        method = 'line-delta'
    build_details = (method, noeol)
    start = start + 1
    raw_record = bytes[start:]
    annotated = 'annotated' in storage_kind
    return [KnitContentFactory(key, parents, build_details, None, raw_record, annotated, network_bytes=bytes)]