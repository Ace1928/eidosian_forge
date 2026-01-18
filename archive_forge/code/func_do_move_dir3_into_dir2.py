import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
def do_move_dir3_into_dir2(self):
    return [('rename', ('dir3', 'dir1/dir2/dir3'))]