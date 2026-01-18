from breezy import ignores, tests
def test_ls_ignored(self):
    self.wt.add(['a', '.bzrignore'])
    self.build_tree(['blah.py', 'blah.pyo', 'user-ignore'])
    self.ls_equals('.bzrignore\na\nblah.py\nblah.pyo\nuser-ignore\n')
    self.ls_equals('V        .bzrignore\nV        a\n?        blah.py\nI        blah.pyo\nI        user-ignore\n', '--verbose')
    self.ls_equals('blah.pyo\nuser-ignore\n', '--ignored')
    self.ls_equals('blah.py\n', '--unknown')
    self.ls_equals('.bzrignore\na\n', '--versioned')
    self.ls_equals('.bzrignore\na\n', '-V')