from __future__ import unicode_literals
import distutils.errors
from distutils import log
import errno
import io
import os
import re
import subprocess
import time
import pkg_resources
from pbr import options
from pbr import version
def get_git_short_sha(git_dir=None):
    """Return the short sha for this repo, if it exists."""
    if not git_dir:
        git_dir = _run_git_functions()
    if git_dir:
        return _run_git_command(['log', '-n1', '--pretty=format:%h'], git_dir)
    return None