import os
import time
from pathlib import Path
from unittest import mock, SkipTest
import pytest
from ...testing import TempFATFS
from ...utils.filemanip import (
def nfs_unlink(pathlike, *, dir_fd=None):
    if dir_fd is None:
        path = Path(pathlike)
        deleted = path.with_name('.nfs00000000')
        path.rename(deleted)
    else:
        os.rename(pathlike, '.nfs1111111111', src_dir_fd=dir_fd, dst_dir_fd=dir_fd)