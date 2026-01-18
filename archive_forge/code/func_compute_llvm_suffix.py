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
def compute_llvm_suffix(coredata: coredata.CoreData):
    compilers = coredata.compilers[MachineChoice.BUILD]
    cpp_compiler_is_clang = 'cpp' in compilers and compilers['cpp'].id == 'clang'
    c_compiler_is_clang = 'c' in compilers and compilers['c'].id == 'clang'
    if cpp_compiler_is_clang:
        suffix = compilers['cpp'].version.split('.')[0]
        return suffix
    if c_compiler_is_clang:
        suffix = compilers['c'].version.split('.')[0]
        return suffix
    return None