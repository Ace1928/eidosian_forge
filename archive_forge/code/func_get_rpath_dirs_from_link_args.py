from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, InitVar
from functools import lru_cache
from itertools import chain
from pathlib import Path
import copy
import enum
import json
import os
import pickle
import re
import shlex
import shutil
import typing as T
import hashlib
from .. import build
from .. import dependencies
from .. import programs
from .. import mesonlib
from .. import mlog
from ..compilers import LANGUAGES_USING_LDFLAGS, detect
from ..mesonlib import (
@staticmethod
def get_rpath_dirs_from_link_args(args: T.List[str]) -> T.Set[str]:
    dirs: T.Set[str] = set()
    rpath_regex = re.compile('-Wl,-rpath[=,]([^,]+)')
    runpath_regex = re.compile('-Wl,-R[,]?([^,]+)')
    symbols_regex = re.compile('-Wl,--just-symbols[=,]([^,]+)')
    for arg in args:
        rpath_match = rpath_regex.match(arg)
        if rpath_match:
            for dir in rpath_match.group(1).split(':'):
                dirs.add(dir)
        runpath_match = runpath_regex.match(arg)
        if runpath_match:
            for dir in runpath_match.group(1).split(':'):
                if Path(dir).is_dir():
                    dirs.add(dir)
        symbols_match = symbols_regex.match(arg)
        if symbols_match:
            for dir in symbols_match.group(1).split(':'):
                if Path(dir).is_dir():
                    raise MesonException(f'Invalid arg for --just-symbols, {dir} is a directory.')
    return dirs