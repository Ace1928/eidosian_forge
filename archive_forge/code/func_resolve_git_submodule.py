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
def resolve_git_submodule(self) -> bool:
    if not GIT:
        return False
    if not os.path.isdir(self.dirname):
        return False
    ret, out = quiet_git(['rev-parse'], Path(self.dirname).parent)
    if not ret:
        return False
    ret, out = quiet_git(['submodule', 'status', '.'], self.dirname)
    if not ret:
        return False
    if out.startswith('+'):
        mlog.warning('git submodule might be out of date')
        return True
    elif out.startswith('U'):
        raise WrapException('git submodule has merge conflicts')
    elif out.startswith('-'):
        if verbose_git(['submodule', 'update', '--init', '.'], self.dirname):
            return True
        raise WrapException('git submodule failed to init')
    elif out.startswith(' '):
        verbose_git(['submodule', 'update', '.'], self.dirname)
        verbose_git(['checkout', '.'], self.dirname)
        return True
    elif out == '':
        return False
    raise WrapException(f'Unknown git submodule output: {out!r}')