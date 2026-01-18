import errno
import os
import shutil
import socket
import tempfile
from ...objects import hex_to_sha
from ...protocol import CAPABILITY_SIDE_BAND_64K
from ...repo import Repo
from ...server import ReceivePackHandler
from ..utils import tear_down_repo
from .utils import require_git_version, run_git_or_fail
def branch_args(self, branches=None):
    if branches is None:
        branches = ['master', 'branch']
    return [f'{b}:{b}' for b in branches]