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
def apply_delta(self, delta, new_version_id):
    """Apply delta to this object to become new_version_id."""
    offset = 0
    lines = self._lines
    for start, end, count, delta_lines in delta:
        lines[offset + start:offset + end] = delta_lines
        offset = offset + (start - end) + count
    self._version_id = new_version_id