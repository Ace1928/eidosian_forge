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
def guess_install_tag(self, fname: str, outdir: T.Optional[str]=None) -> T.Optional[str]:
    prefix = self.environment.get_prefix()
    bindir = Path(prefix, self.environment.get_bindir())
    libdir = Path(prefix, self.environment.get_libdir())
    incdir = Path(prefix, self.environment.get_includedir())
    _ldir = self.environment.coredata.get_option(mesonlib.OptionKey('localedir'))
    assert isinstance(_ldir, str), 'for mypy'
    localedir = Path(prefix, _ldir)
    dest_path = Path(prefix, outdir, Path(fname).name) if outdir else Path(prefix, fname)
    if bindir in dest_path.parents:
        return 'runtime'
    elif libdir in dest_path.parents:
        if dest_path.suffix in {'.a', '.pc'}:
            return 'devel'
        elif dest_path.suffix in {'.so', '.dll'}:
            return 'runtime'
    elif incdir in dest_path.parents:
        return 'devel'
    elif localedir in dest_path.parents:
        return 'i18n'
    elif 'installed-tests' in dest_path.parts:
        return 'tests'
    elif 'systemtap' in dest_path.parts:
        return 'systemtap'
    mlog.debug('Failed to guess install tag for', dest_path)
    return None