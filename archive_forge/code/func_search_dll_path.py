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
@lru_cache(maxsize=None)
def search_dll_path(link_arg: str) -> T.Optional[str]:
    if link_arg.startswith(('-l', '-L')):
        link_arg = link_arg[2:]
    p = Path(link_arg)
    if not p.is_absolute():
        return None
    try:
        p = p.resolve(strict=True)
    except FileNotFoundError:
        return None
    for f in p.parent.glob('*.dll'):
        return str(p.parent)
    if p.is_file():
        p = p.parent
    binpath = Path('/bin'.join(p.as_posix().rsplit('/lib', maxsplit=1)))
    for _ in binpath.glob('*.dll'):
        return str(binpath)
    return None