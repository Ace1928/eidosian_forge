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
def import_tar(tree, tar_input):
    """Replace the contents of a working directory with tarfile contents.
    The tarfile may be a gzipped stream.  File ids will be updated.
    """
    tar_file = tarfile.open('lala', 'r', tar_input)
    import_archive(tree, tar_file)