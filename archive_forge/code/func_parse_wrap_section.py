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
def parse_wrap_section(self, config: configparser.ConfigParser) -> None:
    if len(config.sections()) < 1:
        raise WrapException(f'Missing sections in {self.basename}')
    self.wrap_section = config.sections()[0]
    if not self.wrap_section.startswith('wrap-'):
        raise WrapException(f'{self.wrap_section!r} is not a valid first section in {self.basename}')
    self.type = self.wrap_section[5:]
    self.values = dict(config[self.wrap_section])
    if 'diff_files' in self.values:
        FeatureNew('Wrap files with diff_files', '0.63.0').use(self.subproject)
        for s in self.values['diff_files'].split(','):
            path = Path(s.strip())
            if path.is_absolute():
                raise WrapException('diff_files paths cannot be absolute')
            if '..' in path.parts:
                raise WrapException('diff_files paths cannot contain ".."')
            self.diff_files.append(path)