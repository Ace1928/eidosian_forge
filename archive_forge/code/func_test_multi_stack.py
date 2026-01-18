from breezy import errors, tests, urlutils
from breezy.bzr import remote
from breezy.tests.per_repository import TestCaseWithRepository
def test_multi_stack(self):
    """base + stacked + stacked-on-stacked"""
    base_tree, stacked_tree = self.make_stacked_target()
    self.build_tree(['stacked/f3.txt'])
    stacked_tree.add(['f3.txt'], ids=[b'f3.txt-id'])
    stacked_key = (b'stacked-rev-id',)
    stacked_tree.commit('add f3', rev_id=stacked_key[0])
    stacked_only_repo = self.get_only_repo(stacked_tree)
    self.assertPresent([self.r2_key], stacked_only_repo.inventories, [self.r1_key, self.r2_key])
    stacked2_url = urlutils.join(base_tree.branch.base, '../stacked2')
    stacked2_bzrdir = stacked_tree.controldir.sprout(stacked2_url, revision_id=self.r1_key[0], stacked=True)
    if isinstance(stacked2_bzrdir, remote.RemoteBzrDir):
        stacked2_branch = stacked2_bzrdir.open_branch()
        stacked2_tree = stacked2_branch.create_checkout('stacked2', lightweight=True)
    else:
        stacked2_tree = stacked2_bzrdir.open_workingtree()
    self.build_tree(['stacked2/f3.txt'])
    stacked2_only_repo = self.get_only_repo(stacked2_tree)
    self.assertPresent([], stacked2_only_repo.inventories, [self.r1_key, self.r2_key])
    stacked2_tree.add(['f3.txt'], ids=[b'f3.txt-id'])
    stacked2_tree.commit('add f3', rev_id=b'stacked2-rev-id')
    stacked2_only_repo.refresh_data()
    self.assertPresent([self.r1_key], stacked2_only_repo.inventories, [self.r1_key, self.r2_key])