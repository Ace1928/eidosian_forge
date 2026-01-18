from breezy import branchbuilder, errors, tests, urlutils
from breezy.branch import Branch
from breezy.controldir import BranchReferenceLoop
from breezy.tests import per_controldir
from breezy.tests.features import UnicodeFilenameFeature
def test_branch_reference_loop(self):
    repo = self.make_repository('repo')
    to_branch = self.create_branch(repo.controldir, name='somebranch')
    try:
        self.assertRaises(BranchReferenceLoop, repo.controldir.set_branch_reference, to_branch, name='somebranch')
    except errors.IncompatibleFormat:
        raise tests.TestNotApplicable('Control dir does not support creating branch references.')