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
def print_default_values_warning() -> None:
    mlog.warning('The source directory instead of the build directory was specified.')
    mlog.warning('Only the default values for the project are printed.')