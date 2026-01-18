import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
def test_adopt_child(self):
    self.run_script("\n$ brz mv -q dir/file2 file2\n$ brz rm -q dir --no-backup\n$ brz resolve dir\n2>2 conflicts resolved, 0 remaining\n$ brz commit -q --strict -m 'No more conflicts nor unknown files'\n")