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
def import_repo(self, name):
    """Import a repo from a fast-export file in a temporary directory.

        Args:
          name: The name of the repository export file, relative to
            dulwich/tests/data/repos.
        Returns: An initialized Repo object that lives in a temporary
            directory.
        """
    path = import_repo_to_dir(name)
    repo = Repo(path)

    def cleanup():
        repo.close()
        rmtree_ro(os.path.dirname(path.rstrip(os.sep)))
    self.addCleanup(cleanup)
    return repo