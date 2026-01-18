from breezy import ignores, tests
def test_ls_added(self):
    self.wt.add(['a'])
    self.ls_equals('?        .bzrignore\nV        a\n', '--verbose')
    self.wt.commit('add')
    self.build_tree(['subdir/'])
    self.ls_equals('?        .bzrignore\nV        a\n?        subdir/\n', '--verbose')
    self.build_tree(['subdir/b'])
    self.wt.add(['subdir/', 'subdir/b', '.bzrignore'])
    self.ls_equals('V        .bzrignore\nV        a\nV        subdir/\nV        subdir/b\n', '--verbose')