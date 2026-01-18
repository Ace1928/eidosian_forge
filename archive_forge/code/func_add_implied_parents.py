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
def add_implied_parents(implied_parents, path):
    """Update the set of implied parents from a path"""
    parent = os.path.dirname(path)
    if parent in implied_parents:
        return
    implied_parents.add(parent)
    add_implied_parents(implied_parents, parent)