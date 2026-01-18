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
def git_commit_messages():
    """
    Output each commit message between here and master.
    """
    fork_point = git_.merge_base('origin/master', 'HEAD').strip()
    messages = git_.log(fork_point + '..HEAD')
    return messages