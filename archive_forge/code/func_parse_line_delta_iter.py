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
def parse_line_delta_iter(self, lines, version_id):
    cur = 0
    num_lines = len(lines)
    while cur < num_lines:
        header = lines[cur]
        cur += 1
        start, end, c = (int(n) for n in header.split(b','))
        yield (start, end, c, lines[cur:cur + c])
        cur += c