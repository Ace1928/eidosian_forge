import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
def do_modify_file_A_in_dir(self):
    return [('modify', ('dir/file', b'trunk content\nfeature A\n'))]