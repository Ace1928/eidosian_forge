from breezy import tests, urlutils
from breezy.bzr import remote
from breezy.tests.per_repository import TestCaseWithRepository
def make_double_stacked_branches(self):
    wt_a = self.make_branch_and_tree('a')
    branch_a = wt_a.branch
    branch_b = self.make_branch('b')
    branch_b.set_stacked_on_url(urlutils.relative_url(branch_b.base, branch_a.base))
    branch_c = self.make_branch('c')
    branch_c.set_stacked_on_url(urlutils.relative_url(branch_c.base, branch_b.base))
    revid_1 = wt_a.commit('first commit')
    return (branch_a, branch_b, branch_c, revid_1)