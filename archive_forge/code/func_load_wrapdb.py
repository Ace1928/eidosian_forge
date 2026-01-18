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
def load_wrapdb(self) -> None:
    try:
        with Path(self.subdir_root, 'wrapdb.json').open('r', encoding='utf-8') as f:
            self.wrapdb = json.load(f)
    except FileNotFoundError:
        return
    for name, info in self.wrapdb.items():
        self.wrapdb_provided_deps.update({i: name for i in info.get('dependency_names', [])})
        self.wrapdb_provided_programs.update({i: name for i in info.get('program_names', [])})