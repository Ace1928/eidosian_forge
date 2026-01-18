import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
def check_dir_renamed2(self):
    self.assertPathDoesNotExist('branch/dir')
    self.assertPathExists('branch/new-dir2')