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
def do_directory(tt, trans_id, tree, relative_path, path):
    if isdir(path) and tree.is_versioned(relative_path):
        tt.cancel_deletion(trans_id)
    else:
        tt.create_directory(trans_id)