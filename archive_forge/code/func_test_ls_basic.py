from breezy import ignores, tests
def test_ls_basic(self):
    """Test the abilities of 'brz ls'"""
    self.ls_equals('.bzrignore\na\n')
    self.ls_equals('.bzrignore\na\n', './')
    self.ls_equals('?        .bzrignore\n?        a\n', '--verbose')
    self.ls_equals('.bzrignore\na\n', '--unknown')
    self.ls_equals('', '--ignored')
    self.ls_equals('', '--versioned')
    self.ls_equals('', '-V')
    self.ls_equals('.bzrignore\na\n', '--unknown --ignored --versioned')
    self.ls_equals('.bzrignore\na\n', '--unknown --ignored -V')
    self.ls_equals('', '--ignored --versioned')
    self.ls_equals('', '--ignored -V')
    self.ls_equals('.bzrignore\x00a\x00', '--null')