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
def revision_id_bzr_to_foreign(self, bzr_revid):
    if bzr_revid == NULL_REVISION:
        from dulwich.protocol import ZERO_SHA
        return (ZERO_SHA, None)
    if not bzr_revid.startswith(b'git-'):
        raise errors.InvalidRevisionId(bzr_revid, None)
    mapping_version, git_sha = bzr_revid.split(b':', 1)
    mapping = self.get(mapping_version)
    return mapping.revision_id_bzr_to_foreign(bzr_revid)