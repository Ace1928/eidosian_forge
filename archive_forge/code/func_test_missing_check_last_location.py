from breezy import osutils, tests
def test_missing_check_last_location(self):
    wt = self.make_branch_and_tree('a')
    b = wt.branch
    self.build_tree(['a/foo'])
    wt.add('foo')
    wt.commit('initial')
    location = osutils.getcwd() + '/a/'
    b.controldir.sprout('b')
    lines, err = self.run_bzr('missing', working_dir='b')
    self.assertEqual('Using saved parent location: %s\nBranches are up to date.\n' % location, lines)
    self.assertEqual('', err)