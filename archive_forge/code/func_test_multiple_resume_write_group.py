import sys
from breezy import branch, controldir, errors, memorytree, tests
from breezy.bzr import branch as bzrbranch
from breezy.bzr import remote, versionedfile
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_multiple_resume_write_group(self):
    self.require_suspendable_write_groups('Cannot test resume on repo that does not support suspending')
    repo = self.make_write_locked_repo()
    repo.start_write_group()
    first_key = (b'file-id', b'revid')
    repo.texts.add_lines(first_key, (), [b'lines'])
    wg_tokens = repo.suspend_write_group()
    same_repo = self.reopen_repo(repo)
    same_repo.resume_write_group(wg_tokens)
    self.assertTrue(same_repo.is_in_write_group())
    second_key = (b'file-id', b'second-revid')
    same_repo.texts.add_lines(second_key, (first_key,), [b'more lines'])
    try:
        new_wg_tokens = same_repo.suspend_write_group()
    except:
        same_repo.abort_write_group(suppress_errors=True)
        raise
    self.assertEqual(2, len(new_wg_tokens))
    self.assertSubset(wg_tokens, new_wg_tokens)
    same_repo = self.reopen_repo(repo)
    same_repo.resume_write_group(new_wg_tokens)
    both_keys = {first_key, second_key}
    self.assertEqual(both_keys, same_repo.texts.keys())
    same_repo.abort_write_group()