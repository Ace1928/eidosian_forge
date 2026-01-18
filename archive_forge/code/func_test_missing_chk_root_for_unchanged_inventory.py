from breezy import errors, osutils, repository
from breezy.bzr import btree_index
from breezy.bzr.remote import RemoteRepository
from breezy.bzr.tests.per_repository_chk import TestCaseWithRepositoryCHK
from breezy.bzr.versionedfile import VersionedFiles
from breezy.tests import TestNotApplicable
def test_missing_chk_root_for_unchanged_inventory(self):
    """commit_write_group fails with BzrCheckError when the chk root record
        for a new inventory is missing, even if the parent inventory is present
        and has identical content (i.e. the same chk root).

        A stacked repository containing only a revision with an identical
        inventory to its parent will still have the chk root records for those
        inventories.

        (In principle the chk records are unnecessary in this case, but in
        practice bzr 2.0rc1 (at least) expects to find them.)
        """
    repo = self.make_repository('damaged-repo')
    builder = self.make_branch_builder('simple-branch')
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None)), ('add', ('file', b'file-id', 'file', b'content\n'))], revision_id=b'A-id')
    builder.build_snapshot(None, [], revision_id=b'B-id')
    builder.build_snapshot(None, [], revision_id=b'C-id')
    b = builder.get_branch()
    b.lock_read()
    self.addCleanup(b.unlock)
    inv_b = b.repository.get_inventory(b'B-id')
    inv_c = b.repository.get_inventory(b'C-id')
    if not isinstance(repo, RemoteRepository):
        self.assertEqual(inv_b.id_to_entry.key(), inv_c.id_to_entry.key())
    repo.lock_write()
    repo.start_write_group()
    src_repo = b.repository
    repo.inventories.insert_record_stream(src_repo.inventories.get_record_stream([(b'B-id',), (b'C-id',)], 'unordered', True))
    repo.revisions.insert_record_stream(src_repo.revisions.get_record_stream([(b'C-id',)], 'unordered', True))
    repo.add_fallback_repository(b.repository)
    self.assertRaises(errors.BzrCheckError, repo.commit_write_group)
    reopened_repo = self.reopen_repo_and_resume_write_group(repo)
    self.assertRaises(errors.BzrCheckError, reopened_repo.commit_write_group)
    reopened_repo.abort_write_group()