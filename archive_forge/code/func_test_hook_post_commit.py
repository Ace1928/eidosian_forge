import os
import shutil
import stat
import sys
import tempfile
from dulwich import errors
from dulwich.tests import TestCase
from ..hooks import CommitMsgShellHook, PostCommitShellHook, PreCommitShellHook
def test_hook_post_commit(self):
    fd, path = tempfile.mkstemp()
    os.close(fd)
    repo_dir = os.path.join(tempfile.mkdtemp())
    os.mkdir(os.path.join(repo_dir, 'hooks'))
    self.addCleanup(shutil.rmtree, repo_dir)
    post_commit_success = '#!/bin/sh\nrm ' + path + '\n'
    post_commit_fail = '#!/bin/sh\nexit 1\n'
    post_commit_cwd = '#!/bin/sh\nif [ "$(pwd)" = \'' + repo_dir + "' ]; then exit 0; else exit 1; fi\n"
    post_commit = os.path.join(repo_dir, 'hooks', 'post-commit')
    hook = PostCommitShellHook(repo_dir)
    with open(post_commit, 'w') as f:
        f.write(post_commit_fail)
    os.chmod(post_commit, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
    self.assertRaises(errors.HookError, hook.execute)
    if sys.platform != 'darwin':
        with open(post_commit, 'w') as f:
            f.write(post_commit_cwd)
        os.chmod(post_commit, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        hook.execute()
    with open(post_commit, 'w') as f:
        f.write(post_commit_success)
    os.chmod(post_commit, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
    hook.execute()
    self.assertFalse(os.path.exists(path))