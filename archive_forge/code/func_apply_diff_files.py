from __future__ import annotations
from .. import mlog
import contextlib
from dataclasses import dataclass
import urllib.request
import urllib.error
import urllib.parse
import os
import hashlib
import shutil
import tempfile
import stat
import subprocess
import sys
import configparser
import time
import typing as T
import textwrap
import json
from base64 import b64encode
from netrc import netrc
from pathlib import Path, PurePath
from functools import lru_cache
from . import WrapMode
from .. import coredata
from ..mesonlib import quiet_git, GIT, ProgressBar, MesonException, windows_proof_rmtree, Popen_safe
from ..interpreterbase import FeatureNew
from ..interpreterbase import SubProject
from .. import mesonlib
def apply_diff_files(self) -> None:
    for filename in self.wrap.diff_files:
        mlog.log(f'Applying diff file "{filename}"')
        path = Path(self.wrap.filesdir) / filename
        if not path.exists():
            raise WrapException(f'Diff file "{path}" does not exist')
        relpath = os.path.relpath(str(path), self.dirname)
        if PATCH:
            cmd = [PATCH, '-l', '-f', '-p1', '-i', str(Path(relpath).as_posix())]
        elif GIT:
            cmd = [GIT, '--work-tree', '.', 'apply', '--ignore-whitespace', '-p1', relpath]
        else:
            raise WrapException('Missing "patch" or "git" commands to apply diff files')
        p, out, _ = Popen_safe(cmd, cwd=self.dirname, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            mlog.log(out.strip())
            raise WrapException(f'Failed to apply diff file "{filename}"')