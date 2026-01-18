from breezy.tests.per_repository_reference import \
def test_all_revision_ids_from_repo(self):
    tree = self.make_branch_and_tree('spare')
    revid = tree.commit('one')
    base = self.make_repository('base')
    repo = self.make_referring('referring', base)
    repo.fetch(tree.branch.repository, revid)
    self.assertEqual({revid}, set(repo.all_revision_ids()))