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
def get_compiler_system_lib_dirs(self, for_machine: MachineChoice) -> T.List[str]:
    for comp in self.coredata.compilers[for_machine].values():
        if comp.id == 'clang':
            index = 1
            break
        elif comp.id == 'gcc':
            index = 2
            break
    else:
        return []
    p, out, _ = Popen_safe(comp.get_exelist() + ['-print-search-dirs'])
    if p.returncode != 0:
        raise mesonlib.MesonException('Could not calculate system search dirs')
    out = out.split('\n')[index].lstrip('libraries: =').split(':')
    return [os.path.normpath(p) for p in out]