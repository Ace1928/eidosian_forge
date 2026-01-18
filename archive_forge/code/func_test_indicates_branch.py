from breezy.controldir import ControlDir
from breezy.tests import TestCaseWithTransport
def test_indicates_branch(self):
    t = self.make_repository('a', format='development-colo')
    t.controldir.create_branch(name='another')
    branch = t.controldir.create_branch(name='colocated')
    t.controldir.set_branch_reference(target_branch=branch)
    out, err = self.run_bzr('branches a')
    self.assertEqual(out, '  another\n* colocated\n')