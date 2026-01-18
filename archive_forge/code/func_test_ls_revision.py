from breezy import ignores, tests
def test_ls_revision(self):
    self.wt.add(['a'])
    self.wt.commit('add')
    self.build_tree(['subdir/'])
    self.ls_equals('a\n', '--revision 1')
    self.ls_equals('V        a\n', '--verbose --revision 1')
    self.ls_equals('', '--revision 1', working_dir='subdir')