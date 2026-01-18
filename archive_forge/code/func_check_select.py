import os
from ... import tests
from ...conflicts import resolve
from ...tests import scenarios
from ...tests.test_conflicts import vary_by_conflicts
from .. import conflicts as bzr_conflicts
def check_select(not_selected, selected, paths, **kwargs):
    self.assertEqual((not_selected, selected), tree_conflicts.select_conflicts(tree, paths, **kwargs))