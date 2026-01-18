from breezy.tests import TestCaseWithTransport
from breezy.workingtree import WorkingTree
def test_define_view(self):
    wt = self.make_branch_and_tree('.')
    out, err = self.run_bzr('view a b c')
    self.assertEqual(out, "Using 'my' view: a, b, c\n")
    out, err = self.run_bzr('view e f --name foo')
    self.assertEqual(out, "Using 'foo' view: e, f\n")
    out, err = self.run_bzr('view p q')
    self.assertEqual(out, "Using 'foo' view: p, q\n")
    out, err = self.run_bzr('view r s --name my')
    self.assertEqual(out, "Using 'my' view: r, s\n")
    out, err = self.run_bzr('view a --name off', retcode=3)
    self.assertContainsRe(err, "Cannot change the 'off' pseudo view")