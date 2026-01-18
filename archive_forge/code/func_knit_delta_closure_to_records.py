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
def knit_delta_closure_to_records(storage_kind, bytes, line_end):
    """Convert a network record to a iterator over stream records.

    :param storage_kind: The storage kind of the record.
        Must be 'knit-delta-closure'.
    :param bytes: The bytes of the record on the network.
    """
    generator = _NetworkContentMapGenerator(bytes, line_end)
    return generator.get_record_stream()