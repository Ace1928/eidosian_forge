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
def mode_kind(mode):
    """Determine the Bazaar inventory kind based on Unix file mode."""
    if mode is None:
        return None
    entry_kind = (mode & 229376) / 32768
    if entry_kind == 0:
        return 'directory'
    elif entry_kind == 1:
        file_kind = (mode & 28672) / 4096
        if file_kind == 0:
            return 'file'
        elif file_kind == 2:
            return 'symlink'
        elif file_kind == 6:
            return 'tree-reference'
        else:
            raise AssertionError('Unknown file kind %d, perms=%o.' % (file_kind, mode))
    else:
        raise AssertionError('Unknown kind, perms={!r}.'.format(mode))