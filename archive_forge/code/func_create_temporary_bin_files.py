from __future__ import annotations
import os
import stat
import tarfile
import tempfile
import time
import typing as t
from .constants import (
from .config import (
from .util import (
from .data import (
from .util_common import (
def create_temporary_bin_files(args: CommonConfig) -> tuple[tuple[str, str], ...]:
    """Create a temporary ansible bin directory populated using the symlink map."""
    if args.explain:
        temp_path = '/tmp/ansible-tmp-bin'
    else:
        temp_path = tempfile.mkdtemp(prefix='ansible', suffix='bin')
        ExitHandler.register(remove_tree, temp_path)
        for name, dest in ANSIBLE_BIN_SYMLINK_MAP.items():
            path = os.path.join(temp_path, name)
            os.symlink(dest, path)
    return tuple(((os.path.join(temp_path, name), os.path.join('bin', name)) for name in sorted(ANSIBLE_BIN_SYMLINK_MAP)))