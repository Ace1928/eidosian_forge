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
def escape_file_id(file_id):
    file_id = file_id.replace(b'_', b'__')
    file_id = file_id.replace(b' ', b'_s')
    file_id = file_id.replace(b'\x0c', b'_c')
    return file_id