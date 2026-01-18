import os
from ... import osutils, tests
from ...osutils import canonical_relpath, pathjoin
from .. import KnownFailure
from ..features import CaseInsCasePresFilenameFeature
from ..script import run_script
def test_add_not_found(self):
    """Test add when the input file doesn't exist."""
    wt = self.make_branch_and_tree('.')
    self.build_tree(['MixedCaseParent/', 'MixedCaseParent/MixedCase'])
    expected_fname = pathjoin(wt.basedir, 'MixedCaseParent', 'notfound')
    run_script(self, '\n            $ brz add mixedcaseparent/notfound\n            2>brz: ERROR: No such file: {}\n            '.format(repr(expected_fname)))