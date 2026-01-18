from breezy import errors, osutils, repository
from breezy.bzr import btree_index
from breezy.bzr.remote import RemoteRepository
from breezy.bzr.tests.per_repository_chk import TestCaseWithRepositoryCHK
from breezy.bzr.versionedfile import VersionedFiles
from breezy.tests import TestNotApplicable
def test_missing_chk_root_for_parent_inventory(self):
    """commit_write_group fails with BzrCheckError when the chk root record
        for a parent inventory of a new revision is missing.
        """
    repo = self.make_repository('damaged-repo')
    if isinstance(repo, RemoteRepository):
        raise TestNotApplicable('Unable to obtain CHKInventory from remote repo')
    b = self.make_branch_with_multiple_chk_nodes()
    b.lock_read()
    self.addCleanup(b.unlock)
    inv_c = b.repository.get_inventory(b'C-id')
    chk_keys_for_c_only = [inv_c.id_to_entry.key(), inv_c.parent_id_basename_to_file_id.key()]
    repo.lock_write()
    repo.start_write_group()
    src_repo = b.repository
    repo.chk_bytes.insert_record_stream(src_repo.chk_bytes.get_record_stream(chk_keys_for_c_only, 'unordered', True))
    repo.inventories.insert_record_stream(src_repo.inventories.get_record_stream([(b'B-id',), (b'C-id',)], 'unordered', True))
    repo.revisions.insert_record_stream(src_repo.revisions.get_record_stream([(b'C-id',)], 'unordered', True))
    repo.add_fallback_repository(b.repository)
    self.assertRaises(errors.BzrCheckError, repo.commit_write_group)
    reopened_repo = self.reopen_repo_and_resume_write_group(repo)
    self.assertRaises(errors.BzrCheckError, reopened_repo.commit_write_group)
    reopened_repo.abort_write_group()