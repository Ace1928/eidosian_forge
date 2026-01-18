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
def get_sha1s(self, keys):
    """See VersionedFiles.get_sha1s()."""
    missing = set(keys)
    record_map = self._get_record_map(missing, allow_missing=True)
    result = {}
    for key, details in record_map.items():
        if key not in missing:
            continue
        result[key] = details[2]
    missing.difference_update(set(result))
    for source in self._immediate_fallback_vfs:
        if not missing:
            break
        new_result = source.get_sha1s(missing)
        result.update(new_result)
        missing.difference_update(set(new_result))
    return result