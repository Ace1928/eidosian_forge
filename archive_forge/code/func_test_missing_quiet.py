from breezy import osutils, tests
def test_missing_quiet(self):
    a_tree = self.make_branch_and_tree('.')
    self.build_tree_contents([('a', b'initial\n')])
    a_tree.add('a')
    a_tree.commit(message='initial')
    out, err = self.run_bzr('missing -q .')
    self.assertEqual('', out)
    self.assertEqual('', err)