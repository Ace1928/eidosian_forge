import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
def assertConflict(self, wt):
    confs = wt.conflicts()
    self.assertLength(1, confs)
    c = confs[0]
    self.assertIsInstance(c, self._conflict_type)
    self._assert_conflict(wt, c)