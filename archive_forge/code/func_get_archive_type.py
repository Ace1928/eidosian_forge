import errno
import os
import re
import stat
import tarfile
import zipfile
from io import BytesIO
from . import urlutils
from .bzr import generate_ids
from .controldir import ControlDir, is_control_filename
from .errors import BzrError, CommandError, NotBranchError
from .osutils import (basename, file_iterator, file_kind, isdir, pathjoin,
from .trace import warning
from .transform import resolve_conflicts
from .transport import NoSuchFile, get_transport
from .workingtree import WorkingTree
def get_archive_type(path):
    """Return the type of archive and compressor indicated by path name.

    Only external compressors are returned, so zip files are only
    ('zip', None).  .tgz is treated as ('tar', 'gz') and '.tar.xz' is treated
    as ('tar', 'lzma').
    """
    matches = re.match('.*\\.(zip|tgz|tar(.(gz|bz2|lzma|xz))?)$', path)
    if not matches:
        raise NotArchiveType(path)
    external_compressor = None
    if matches.group(3) is not None:
        archive = 'tar'
        external_compressor = matches.group(3)
        if external_compressor == 'xz':
            external_compressor = 'lzma'
    elif matches.group(1) == 'tgz':
        return ('tar', 'gz')
    else:
        archive = matches.group(1)
    return (archive, external_compressor)