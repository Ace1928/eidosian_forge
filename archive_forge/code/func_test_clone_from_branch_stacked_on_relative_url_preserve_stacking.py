from breezy import branch as _mod_branch
from breezy import check, controldir, errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, fixtures, transport_util
from breezy.tests.per_branch import TestCaseWithBranch
def test_clone_from_branch_stacked_on_relative_url_preserve_stacking(self):
    try:
        stacked_bzrdir = self.make_stacked_bzrdir(in_directory='dir')
    except unstackable_format_errors as e:
        raise TestNotApplicable(e)
    stacked_bzrdir.open_branch().set_stacked_on_url('../stacked-on')
    cloned_bzrdir = stacked_bzrdir.clone(self.get_url('cloned'), preserve_stacking=True)
    self.assertEqual('../dir/stacked-on', cloned_bzrdir.open_branch().get_stacked_on_url())