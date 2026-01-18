from breezy import errors, osutils, repository
from breezy.bzr import btree_index
from breezy.bzr.remote import RemoteRepository
from breezy.bzr.tests.per_repository_chk import TestCaseWithRepositoryCHK
from breezy.bzr.versionedfile import VersionedFiles
from breezy.tests import TestNotApplicable
def test_chk_bytes_are_fully_buffered(self):
    repo = self.make_repository('.')
    repo.lock_write()
    self.addCleanup(repo.unlock)
    with repository.WriteGroup(repo):
        sha1, len, _ = repo.chk_bytes.add_lines((None,), None, [b'foo\n', b'bar\n'], random_id=True)
        self.assertEqual(b'4e48e2c9a3d2ca8a708cb0cc545700544efb5021', sha1)
        self.assertEqual({(b'sha1:4e48e2c9a3d2ca8a708cb0cc545700544efb5021',)}, repo.chk_bytes.keys())
    index = repo.chk_bytes._index._graph_index._indices[0]
    self.assertIsInstance(index, btree_index.BTreeGraphIndex)
    self.assertIs(type(index._leaf_node_cache), dict)
    repo2 = repository.Repository.open(self.get_url())
    repo2.lock_read()
    self.addCleanup(repo2.unlock)
    index = repo2.chk_bytes._index._graph_index._indices[0]
    self.assertIsInstance(index, btree_index.BTreeGraphIndex)
    self.assertIs(type(index._leaf_node_cache), dict)