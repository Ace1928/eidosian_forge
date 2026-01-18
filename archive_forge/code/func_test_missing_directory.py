from breezy import osutils, tests
def test_missing_directory(self):
    """Test --directory option"""
    a_tree = self.make_branch_and_tree('a')
    self.build_tree_contents([('a/a', b'initial\n')])
    a_tree.add('a')
    a_tree.commit(message='initial')
    b_tree = a_tree.controldir.sprout('b').open_workingtree()
    self.build_tree_contents([('b/a', b'initial\nmore\n')])
    b_tree.commit(message='more')
    out2, err2 = self.run_bzr('missing --directory a b', retcode=1)
    out1, err1 = self.run_bzr('missing ../b', retcode=1, working_dir='a')
    self.assertEqualDiff(out1, out2)
    self.assertEqualDiff(err1, err2)