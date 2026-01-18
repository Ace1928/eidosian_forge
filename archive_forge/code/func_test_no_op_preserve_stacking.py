from breezy import branch as _mod_branch
from breezy import check, controldir, errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, fixtures, transport_util
from breezy.tests.per_branch import TestCaseWithBranch
def test_no_op_preserve_stacking(self):
    """With no stacking, preserve_stacking should be a no-op."""
    branch = self.make_branch('source')
    cloned_bzrdir = branch.controldir.clone('cloned', preserve_stacking=True)
    self.assertRaises((errors.NotStacked, _mod_branch.UnstackableBranchFormat), cloned_bzrdir.open_branch().get_stacked_on_url)