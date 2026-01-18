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
def detect_ninja(version: str='1.8.2', log: bool=False) -> T.List[str]:
    r = detect_ninja_command_and_version(version, log)
    return r[0] if r else None