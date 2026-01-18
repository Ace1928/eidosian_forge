import sys
from breezy import branch, controldir, errors, memorytree, tests
from breezy.bzr import branch as bzrbranch
from breezy.bzr import remote, versionedfile
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_commit_resumed_write_group_with_missing_parents(self):
    self.require_suspendable_write_groups('Cannot test resume on repo that does not support suspending')
    source_repo = self.make_source_with_delta_record()
    key_base = (b'file-id', b'base')
    key_delta = (b'file-id', b'delta')
    repo = self.make_write_locked_repo()
    repo.start_write_group()
    stream = source_repo.texts.get_record_stream([key_delta], 'unordered', False)
    repo.texts.insert_record_stream(stream)
    try:
        repo.commit_write_group()
    except errors.BzrCheckError:
        pass
    else:
        same_repo = self.reopen_repo(repo)
        same_repo.lock_read()
        record = next(same_repo.texts.get_record_stream([key_delta], 'unordered', True))
        self.assertEqual(b'more\nlines\n', record.get_bytes_as('fulltext'))
        return
    wg_tokens = repo.suspend_write_group()
    same_repo = self.reopen_repo(repo)
    same_repo.resume_write_group(wg_tokens)
    self.assertRaises(errors.BzrCheckError, same_repo.commit_write_group)
    same_repo.abort_write_group()