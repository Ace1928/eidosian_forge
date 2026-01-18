import platform
import re
from io import StringIO
from .. import tests, version, workingtree
from .scenarios import load_tests_apply_scenarios
def test_get_brz_source_tree(self):
    """Get tree for bzr source, if any."""
    self.permit_source_tree_branch_repo()
    src_tree = version._get_brz_source_tree()
    if src_tree is None:
        raise tests.TestSkipped("bzr tests aren't run from a bzr working tree")
    else:
        self.assertIsInstance(src_tree, workingtree.WorkingTree)