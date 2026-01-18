from __future__ import annotations
import itertools
import os, platform, re, sys, shutil
import typing as T
import collections
from . import coredata
from . import mesonlib
from .mesonlib import (
from . import mlog
from .programs import ExternalProgram
from .envconfig import (
from . import compilers
from .compilers import (
from functools import lru_cache
from mesonbuild import envconfig
@staticmethod
def get_build_command(unbuffered: bool=False) -> T.List[str]:
    cmd = mesonlib.get_meson_command()
    if cmd is None:
        raise MesonBugException('No command?')
    cmd = cmd.copy()
    if unbuffered and 'python' in os.path.basename(cmd[0]):
        cmd.insert(1, '-u')
    return cmd