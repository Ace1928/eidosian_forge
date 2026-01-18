from breezy import ignores, tests
def test_ls_directory(self):
    """Test --directory option"""
    self.wt = self.make_branch_and_tree('dir')
    self.build_tree(['dir/sub/', 'dir/sub/file'])
    self.wt.add(['sub', 'sub/file'])
    self.wt.commit('commit')
    self.ls_equals('sub/\nsub/file\n', '--directory=dir')
    self.ls_equals('sub/file\n', '-d dir sub')