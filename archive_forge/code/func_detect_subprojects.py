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
def detect_subprojects(spdir_name: str, current_dir: str='', result: T.Optional[T.Dict[str, T.List[str]]]=None) -> T.Dict[str, T.List[str]]:
    if result is None:
        result = {}
    spdir = os.path.join(current_dir, spdir_name)
    if not os.path.exists(spdir):
        return result
    for trial in glob(os.path.join(spdir, '*')):
        basename = os.path.basename(trial)
        if trial == 'packagecache':
            continue
        append_this = True
        if os.path.isdir(trial):
            detect_subprojects(spdir_name, trial, result)
        elif trial.endswith('.wrap') and os.path.isfile(trial):
            basename = os.path.splitext(basename)[0]
        else:
            append_this = False
        if append_this:
            if basename in result:
                result[basename].append(trial)
            else:
                result[basename] = [trial]
    return result