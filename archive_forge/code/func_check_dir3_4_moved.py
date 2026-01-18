import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
def check_dir3_4_moved(self):
    self.assertPathDoesNotExist('branch/dir3')
    self.assertPathExists('branch/dir1/dir2/dir3')
    self.assertPathExists('branch/dir1/dir2/dir3/dir4')