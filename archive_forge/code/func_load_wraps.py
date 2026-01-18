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
def load_wraps(self) -> None:
    if not os.path.isdir(self.subdir_root):
        return
    root, dirs, files = next(os.walk(self.subdir_root))
    ignore_dirs = {'packagecache', 'packagefiles'}
    for i in files:
        if not i.endswith('.wrap'):
            continue
        fname = os.path.join(self.subdir_root, i)
        wrap = PackageDefinition(fname, self.subproject)
        self.wraps[wrap.name] = wrap
        ignore_dirs |= {wrap.directory, wrap.name}
    for i in dirs:
        if i in ignore_dirs:
            continue
        fname = os.path.join(self.subdir_root, i)
        wrap = PackageDefinition(fname, self.subproject)
        self.wraps[wrap.name] = wrap
    for wrap in self.wraps.values():
        self.add_wrap(wrap)