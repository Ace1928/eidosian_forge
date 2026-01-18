import errno
import os
import shutil
from contextlib import ExitStack
from typing import List, Optional
from .clean_tree import iter_deletables
from .errors import BzrError, DependencyNotPresent
from .osutils import is_inside
from .trace import warning
from .transform import revert
from .transport import NoSuchFile
from .tree import Tree
from .workingtree import WorkingTree
def relevant(p, t):
    if not p:
        return False
    if not is_inside(subpath, p):
        return False
    if t.is_ignored(p):
        return False
    try:
        if not t.has_versioned_directories() and t.kind(p) == 'directory':
            return False
    except NoSuchFile:
        return True
    return True