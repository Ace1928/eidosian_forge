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
def get_from_wrapdb(self, subp_name: str) -> T.Optional[PackageDefinition]:
    info = self.wrapdb.get(subp_name)
    if not info:
        return None
    self.check_can_download()
    latest_version = info['versions'][0]
    version, revision = latest_version.rsplit('-', 1)
    url = urllib.request.urlopen(f'https://wrapdb.mesonbuild.com/v2/{subp_name}_{version}-{revision}/{subp_name}.wrap')
    fname = Path(self.subdir_root, f'{subp_name}.wrap')
    with fname.open('wb') as f:
        f.write(url.read())
    mlog.log(f'Installed {subp_name} version {version} revision {revision}')
    wrap = PackageDefinition(str(fname))
    self.wraps[wrap.name] = wrap
    self.add_wrap(wrap)
    return wrap