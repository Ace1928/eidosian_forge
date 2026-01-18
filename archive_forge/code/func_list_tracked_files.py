from fnmatch import fnmatch
import os
from pathlib import Path
import re
import subprocess
import pytest
import cartopy
@staticmethod
def list_tracked_files():
    """
        Return a list of all the files under git.

        .. note::

            This function raises a ValueError if the repo root does
            not have a ".git" folder. If git is not installed on the system,
            or cannot be found by subprocess, an IOError may also be raised.

        """
    if not (REPO_DIR / '.git').is_dir():
        raise ValueError(f'{REPO_DIR} is not a git repository.')
    output = subprocess.check_output(['git', 'ls-tree', '-z', '-r', '--name-only', 'HEAD'], cwd=REPO_DIR)
    output = output.rstrip(b'\x00').split(b'\x00')
    res = [fname.decode() for fname in output]
    return res