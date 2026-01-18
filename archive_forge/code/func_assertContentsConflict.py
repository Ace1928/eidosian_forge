import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
def assertContentsConflict(self, wt, c):
    self.assertEqual(self._file_id, c.file_id)
    self.assertEqual(self._path, c.path)