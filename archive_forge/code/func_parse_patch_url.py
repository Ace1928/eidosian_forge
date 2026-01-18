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
def parse_patch_url(patch_url: str) -> T.Tuple[str, str]:
    u = urllib.parse.urlparse(patch_url)
    if u.netloc != 'wrapdb.mesonbuild.com':
        raise WrapException(f'URL {patch_url} does not seems to be a wrapdb patch')
    arr = u.path.strip('/').split('/')
    if arr[0] == 'v1':
        return (arr[-3], arr[-2])
    elif arr[0] == 'v2':
        tag = arr[-2]
        _, version = tag.rsplit('_', 1)
        version, revision = version.rsplit('-', 1)
        return (version, revision)
    else:
        raise WrapException(f'Invalid wrapdb URL {patch_url}')