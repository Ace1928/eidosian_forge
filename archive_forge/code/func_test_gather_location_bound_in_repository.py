import sys
from .. import branch as _mod_branch
from .. import controldir, errors, info
from .. import repository as _mod_repository
from .. import tests, workingtree
from ..bzr import branch as _mod_bzrbranch
def test_gather_location_bound_in_repository(self):
    repo = self.make_repository('repo', shared=True)
    repo.set_make_working_trees(False)
    branch = self.make_branch('branch')
    bound_branch = controldir.ControlDir.create_branch_convenience('repo/bound_branch')
    try:
        bound_branch.bind(branch)
    except _mod_branch.BindingUnsupported:
        raise tests.TestNotApplicable('format does not support bound branches')
    self.assertEqual([('shared repository', bound_branch.repository.controldir.user_url), ('repository branch', bound_branch.controldir.user_url), ('bound to branch', branch.controldir.user_url)], info.gather_location_info(bound_branch.repository, bound_branch))