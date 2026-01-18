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
def canonicalize_filename(fname: str) -> str:
    parts = Path(fname).parts
    hashed = ''
    if len(parts) > 5:
        temp = '/'.join(parts[-5:])
        if len(fname) > len(temp) + 41:
            hashed = hashlib.sha1(fname.encode('utf-8')).hexdigest() + '_'
            fname = temp
    for ch in ('/', '\\', ':'):
        fname = fname.replace(ch, '_')
    return hashed + fname