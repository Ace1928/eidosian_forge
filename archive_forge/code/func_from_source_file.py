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
@staticmethod
@lru_cache(maxsize=None)
def from_source_file(source_root: str, subdir: str, fname: str) -> 'File':
    if not os.path.isfile(os.path.join(source_root, subdir, fname)):
        raise MesonException(f'File {fname} does not exist.')
    return File(False, subdir, fname)