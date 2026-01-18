import os
from ....branch import Branch
from ....tests.blackbox import ExternalBase
def test_always_rebase_merges(self):
    trunk = self.make_branch_and_tree('trunk')
    trunk.commit('base')
    feature2 = trunk.controldir.sprout('feature2').open_workingtree()
    revid2 = feature2.commit('change')
    feature = trunk.controldir.sprout('feature').open_workingtree()
    feature.commit('change')
    feature.merge_from_branch(feature2.branch)
    feature.commit('merge')
    feature.commit('change2')
    trunk.commit('additional upstream change')
    self.run_bzr('rebase --always-rebase-merges ../trunk', working_dir='feature')
    repo = feature.branch.repository
    repo.lock_read()
    self.addCleanup(repo.unlock)
    tip = feature.last_revision()
    merge_id = repo.get_graph().get_parent_map([tip])[tip][0]
    merge_parents = repo.get_graph().get_parent_map([merge_id])[merge_id]
    self.assertEqual(self.strip_last_revid_part(revid2), self.strip_last_revid_part(merge_parents[1]))