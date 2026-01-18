from breezy.tests.per_repository_reference import \
def test_all_revision_ids_empty(self):
    base = self.make_repository('base')
    repo = self.make_referring('referring', base)
    self.assertEqual(set(), set(repo.all_revision_ids()))