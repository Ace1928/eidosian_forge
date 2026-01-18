from __future__ import annotations
import itertools
import shutil
import os
import textwrap
import typing as T
import collections
from . import build
from . import coredata
from . import environment
from . import mesonlib
from . import mintro
from . import mlog
from .ast import AstIDGenerator, IntrospectionInterpreter
from .mesonlib import MachineChoice, OptionKey
def print_nondefault_buildtype_options(self) -> None:
    mismatching = self.coredata.get_nondefault_buildtype_args()
    if not mismatching:
        return
    mlog.log('\nThe following option(s) have a different value than the build type default\n')
    mlog.log('               current   default')
    for m in mismatching:
        mlog.log(f'{m[0]:21}{m[1]:10}{m[2]:10}')