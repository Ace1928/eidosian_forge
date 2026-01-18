from breezy import ignores, tests
def test_ls_branch(self):
    """If a branch is specified, files are listed from it"""
    self.build_tree(['subdir/', 'subdir/b'])
    self.wt.add(['subdir', 'subdir/b'])
    self.wt.commit('committing')
    branch = self.make_branch('branchdir')
    branch.pull(self.wt.branch)
    self.ls_equals('branchdir/subdir/\nbranchdir/subdir/b\n', 'branchdir')
    self.ls_equals('branchdir/subdir/\nbranchdir/subdir/b\n', 'branchdir --revision 1')