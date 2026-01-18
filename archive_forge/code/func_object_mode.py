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
def object_mode(kind, executable):
    if kind == 'directory':
        return stat.S_IFDIR
    elif kind == 'symlink':
        mode = stat.S_IFLNK
        if executable:
            mode |= 73
        return mode
    elif kind == 'file':
        mode = stat.S_IFREG | 420
        if executable:
            mode |= 73
        return mode
    elif kind == 'tree-reference':
        from dulwich.objects import S_IFGITLINK
        return S_IFGITLINK
    else:
        raise AssertionError