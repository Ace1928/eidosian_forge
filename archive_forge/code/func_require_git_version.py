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
def require_git_version(required_version, git_path=_DEFAULT_GIT):
    """Require git version >= version, or skip the calling test.

    Args:
      required_version: A tuple of ints of the form (major, minor, point,
        sub-point); omitted components default to 0.
      git_path: Path to the git executable; defaults to the version in
        the system path.

    Raises:
      ValueError: if the required version tuple has too many parts.
      SkipTest: if no suitable git version was found at the given path.
    """
    found_version = git_version(git_path=git_path)
    if found_version is None:
        raise SkipTest(f'Test requires git >= {required_version}, but c git not found')
    if len(required_version) > _VERSION_LEN:
        raise ValueError('Invalid version tuple %s, expected %i parts' % (required_version, _VERSION_LEN))
    required_version = list(required_version)
    while len(found_version) < len(required_version):
        required_version.append(0)
    required_version = tuple(required_version)
    if found_version < required_version:
        required_version = '.'.join(map(str, required_version))
        found_version = '.'.join(map(str, found_version))
        raise SkipTest(f'Test requires git >= {required_version}, found {found_version}')