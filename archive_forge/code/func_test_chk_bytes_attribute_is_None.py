from breezy.bzr.tests.per_repository_chk import TestCaseWithRepositoryCHK
def test_chk_bytes_attribute_is_None(self):
    repo = self.make_repository('.')
    self.assertEqual(None, repo.chk_bytes)