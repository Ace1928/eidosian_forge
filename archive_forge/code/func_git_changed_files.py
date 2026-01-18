import os
import unittest
import contextlib
import tempfile
import shutil
import io
import signal
from typing import Tuple, Dict, Any
from parlai.core.opt import Opt
import parlai.utils.logging as logging
def git_changed_files(skip_nonexisting=True):
    """
    List all the changed files in the git repository.

    :param bool skip_nonexisting:
        If true, ignore files that don't exist on disk. This is useful for
        disregarding files created in master, but don't exist in HEAD.
    """
    fork_point = git_.merge_base('origin/master', 'HEAD').strip()
    filenames = git_.diff('--name-only', fork_point).split('\n')
    if skip_nonexisting:
        filenames = [fn for fn in filenames if os.path.exists(fn)]
    return filenames