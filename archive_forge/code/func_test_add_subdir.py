import os
from ... import osutils, tests
from ...osutils import canonical_relpath, pathjoin
from .. import KnownFailure
from ..features import CaseInsCasePresFilenameFeature
from ..script import run_script
def test_add_subdir(self):
    """test_add_simple but with subdirectories tested too."""
    wt = self.make_branch_and_tree('.')
    self.build_tree(['CamelCaseParent/', 'CamelCaseParent/CamelCase'])
    run_script(self, '\n            $ brz add camelcaseparent/camelcase\n            adding CamelCaseParent\n            adding CamelCaseParent/CamelCase\n            ')