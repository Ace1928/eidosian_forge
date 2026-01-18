from breezy import ignores, tests
def test_kinds(self):
    self.build_tree(['subdir/'])
    self.ls_equals('.bzrignore\na\n', '--kind=file')
    self.ls_equals('subdir/\n', '--kind=directory')
    self.ls_equals('', '--kind=symlink')
    self.run_bzr_error(['invalid kind specified'], 'ls --kind=pile')