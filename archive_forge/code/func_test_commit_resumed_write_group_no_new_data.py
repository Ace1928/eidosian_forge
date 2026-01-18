import sys
from breezy import branch, controldir, errors, memorytree, tests
from breezy.bzr import branch as bzrbranch
from breezy.bzr import remote, versionedfile
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_commit_resumed_write_group_no_new_data(self):
    self.require_suspendable_write_groups('Cannot test resume on repo that does not support suspending')
    repo = self.make_write_locked_repo()
    repo.start_write_group()
    text_key = (b'file-id', b'revid')
    repo.texts.add_lines(text_key, (), [b'lines'])
    wg_tokens = repo.suspend_write_group()
    same_repo = self.reopen_repo(repo)
    same_repo.resume_write_group(wg_tokens)
    same_repo.commit_write_group()
    self.assertEqual([text_key], list(same_repo.texts.keys()))
    self.assertEqual(b'lines', next(same_repo.texts.get_record_stream([text_key], 'unordered', True)).get_bytes_as('fulltext'))
    self.assertRaises(errors.UnresumableWriteGroup, same_repo.resume_write_group, wg_tokens)