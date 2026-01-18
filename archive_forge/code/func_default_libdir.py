from __future__ import annotations
from pathlib import Path
import argparse
import enum
import sys
import stat
import time
import abc
import platform, subprocess, operator, os, shlex, shutil, re
import collections
from functools import lru_cache, wraps, total_ordering
from itertools import tee
from tempfile import TemporaryDirectory, NamedTemporaryFile
import typing as T
import textwrap
import pickle
import errno
import json
from mesonbuild import mlog
from .core import MesonException, HoldableObject
from glob import glob
def default_libdir() -> str:
    if is_debianlike():
        try:
            pc = subprocess.Popen(['dpkg-architecture', '-qDEB_HOST_MULTIARCH'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            stdo, _ = pc.communicate()
            if pc.returncode == 0:
                archpath = stdo.decode().strip()
                return 'lib/' + archpath
        except Exception:
            pass
    if is_freebsd() or is_irix():
        return 'lib'
    if os.path.isdir('/usr/lib64') and (not os.path.islink('/usr/lib64')):
        return 'lib64'
    return 'lib'