import sys
from breezy import branch, controldir, errors, memorytree, tests
from breezy.bzr import branch as bzrbranch
from breezy.bzr import remote, versionedfile
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_ghost_revision(self):
    """A parent inventory may be absent if all the needed texts are present.
        i.e., a ghost revision isn't (necessarily) considered to be a missing
        parent inventory.
        """
    trunk_repo = self.make_stackable_repo()
    self.make_first_commit(trunk_repo)
    trunk_repo.lock_read()
    self.addCleanup(trunk_repo.unlock)
    branch_repo = self.make_new_commit_in_new_repo(trunk_repo, parents=[b'rev-1', b'ghost-rev'])
    inv = branch_repo.get_inventory(b'rev-2')
    repo = self.make_stackable_repo('stacked')
    repo.lock_write()
    repo.start_write_group()
    rich_root = branch_repo._format.rich_root_data
    all_texts = [(ie.file_id, ie.revision) for ie in inv.iter_just_entries() if rich_root or inv.id2path(ie.file_id) != '']
    repo.texts.insert_record_stream(branch_repo.texts.get_record_stream(all_texts, 'unordered', False))
    repo.add_inventory(b'rev-2', inv, [b'rev-1', b'ghost-rev'])
    repo.revisions.insert_record_stream(branch_repo.revisions.get_record_stream([(b'rev-2',)], 'unordered', False))
    self.assertEqual(set(), repo.get_missing_parent_inventories())
    reopened_repo = self.reopen_repo_and_resume_write_group(repo)
    self.assertEqual(set(), reopened_repo.get_missing_parent_inventories())
    reopened_repo.abort_write_group()