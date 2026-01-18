import errno
import functools
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
from typing import Tuple
from dulwich.tests import SkipTest, TestCase
from ...protocol import TCP_GIT_PORT
from ...repo import Repo
def import_repo_to_dir(name):
    """Import a repo from a fast-export file in a temporary directory.

    These are used rather than binary repos for compat tests because they are
    more compact and human-editable, and we already depend on git.

    Args:
      name: The name of the repository export file, relative to
        dulwich/tests/data/repos.
    Returns: The path to the imported repository.
    """
    temp_dir = tempfile.mkdtemp()
    export_path = os.path.join(_REPOS_DATA_DIR, name)
    temp_repo_dir = os.path.join(temp_dir, name)
    export_file = open(export_path, 'rb')
    run_git_or_fail(['init', '--quiet', '--bare', temp_repo_dir])
    run_git_or_fail(['fast-import'], input=export_file.read(), cwd=temp_repo_dir)
    export_file.close()
    return temp_repo_dir