import os
import shutil
import stat
import sys
import tempfile
from dulwich import errors
from dulwich.tests import TestCase
from ..hooks import CommitMsgShellHook, PostCommitShellHook, PreCommitShellHook
def test_hook_pre_commit(self):
    repo_dir = os.path.join(tempfile.mkdtemp())
    os.mkdir(os.path.join(repo_dir, 'hooks'))
    self.addCleanup(shutil.rmtree, repo_dir)
    pre_commit_fail = '#!/bin/sh\nexit 1\n'
    pre_commit_success = '#!/bin/sh\nexit 0\n'
    pre_commit_cwd = '#!/bin/sh\nif [ "$(pwd)" != \'' + repo_dir + '\' ]; then\n    echo "Expected path \'' + repo_dir + '\', got \'$(pwd)\'"\n    exit 1\nfi\n\nexit 0\n'
    pre_commit = os.path.join(repo_dir, 'hooks', 'pre-commit')
    hook = PreCommitShellHook(repo_dir, repo_dir)
    with open(pre_commit, 'w') as f:
        f.write(pre_commit_fail)
    os.chmod(pre_commit, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
    self.assertRaises(errors.HookError, hook.execute)
    if sys.platform != 'darwin':
        with open(pre_commit, 'w') as f:
            f.write(pre_commit_cwd)
        os.chmod(pre_commit, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        hook.execute()
    with open(pre_commit, 'w') as f:
        f.write(pre_commit_success)
    os.chmod(pre_commit, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
    hook.execute()