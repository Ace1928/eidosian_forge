import os
from ... import osutils, tests
from ...osutils import canonical_relpath, pathjoin
from .. import KnownFailure
from ..features import CaseInsCasePresFilenameFeature
from ..script import run_script
def test_add_implied(self):
    """test add with no args sees the correct names."""
    wt = self.make_branch_and_tree('.')
    self.build_tree(['CamelCaseParent/', 'CamelCaseParent/CamelCase'])
    run_script(self, '\n            $ brz add\n            adding CamelCaseParent\n            adding CamelCaseParent/CamelCase\n            ')