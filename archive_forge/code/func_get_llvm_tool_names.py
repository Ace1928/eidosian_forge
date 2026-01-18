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
def get_llvm_tool_names(tool: str) -> T.List[str]:
    suffixes = ['', '-18', '18', '-17', '17', '-16', '16', '-15', '15', '-14', '14', '-13', '13', '-12', '12', '-11', '11', '-10', '10', '-9', '90', '-8', '80', '-7', '70', '-6.0', '60', '-5.0', '50', '-4.0', '40', '-3.9', '39', '-3.8', '38', '-3.7', '37', '-3.6', '36', '-3.5', '35', '-15', '-devel']
    names: T.List[str] = []
    for suffix in suffixes:
        names.append(tool + suffix)
    return names