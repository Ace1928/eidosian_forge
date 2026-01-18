import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
def do_create_dir1_4(self):
    return [('add', ('dir1', b'dir1-id', 'directory', '')), ('add', ('dir1/dir2', b'dir2-id', 'directory', '')), ('add', ('dir3', b'dir3-id', 'directory', '')), ('add', ('dir3/dir4', b'dir4-id', 'directory', ''))]