import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
def do_modify_and_rename_file(self):
    return [('modify', ('new-file', b'trunk content\nmore content\n')), ('rename', ('file', 'new-file'))]