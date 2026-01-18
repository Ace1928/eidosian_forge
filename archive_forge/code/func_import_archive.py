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
def import_archive(tree, archive_file):
    with tree.transform() as tt:
        import_archive_to_transform(tree, archive_file, tt)
        tt.apply()