import base64
import stat
from typing import Optional
import fastbencode as bencode
from .. import errors, foreign, trace, urlutils
from ..foreign import ForeignRevision, ForeignVcs, VcsMappingRegistry
from ..revision import NULL_REVISION, Revision
from .errors import NoPushSupport
from .hg import extract_hg_metadata, format_hg_metadata
from .roundtrip import (CommitSupplement, extract_bzr_metadata,
def parse_file_id(self, file_id):
    if file_id == ROOT_ID:
        return ''
    if not file_id.startswith(FILE_ID_PREFIX):
        raise ValueError
    return decode_git_path(unescape_file_id(file_id[len(FILE_ID_PREFIX):]))