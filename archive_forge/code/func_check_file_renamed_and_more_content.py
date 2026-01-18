import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
def check_file_renamed_and_more_content(self):
    self.assertFileEqual(b'trunk content\nmore content\n', 'branch/new-file')