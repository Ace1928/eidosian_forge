import os
import subprocess
import contextlib
import functools
import tempfile
import shutil
import operator
import warnings
@contextlib.contextmanager
def repo_context(url, branch=None, quiet=True, dest_ctx=temp_dir):
    """
    Check out the repo indicated by url.

    If dest_ctx is supplied, it should be a context manager
    to yield the target directory for the check out.
    """
    exe = 'git' if 'git' in url else 'hg'
    with dest_ctx() as repo_dir:
        cmd = [exe, 'clone', url, repo_dir]
        if branch:
            cmd.extend(['--branch', branch])
        devnull = open(os.path.devnull, 'w')
        stdout = devnull if quiet else None
        subprocess.check_call(cmd, stdout=stdout)
        yield repo_dir