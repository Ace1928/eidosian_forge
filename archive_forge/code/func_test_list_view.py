from breezy.tests import TestCaseWithTransport
from breezy.workingtree import WorkingTree
def test_list_view(self):
    wt = self.make_branch_and_tree('.')
    out, err = self.run_bzr('view')
    self.assertEqual(out, 'No current view.\n')
    self.run_bzr('view a b c')
    out, err = self.run_bzr('view')
    self.assertEqual(out, "'my' view is: a, b, c\n")
    self.run_bzr('view e f --name foo')
    out, err = self.run_bzr('view --name my')
    self.assertEqual(out, "'my' view is: a, b, c\n")
    out, err = self.run_bzr('view --name foo')
    self.assertEqual(out, "'foo' view is: e, f\n")
    out, err = self.run_bzr('view --all')
    self.assertEqual(out.splitlines(), ['Views defined:', '=> foo                  e, f', '   my                   a, b, c'])
    out, err = self.run_bzr('view --name bar', retcode=3)
    self.assertContainsRe(err, 'No such view')