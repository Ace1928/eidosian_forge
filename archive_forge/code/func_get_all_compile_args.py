from __future__ import annotations
import copy
import os
import collections
import itertools
import typing as T
from enum import Enum
from .. import mlog, mesonlib
from ..compilers import clib_langs
from ..mesonlib import LibType, MachineChoice, MesonException, HoldableObject, OptionKey
from ..mesonlib import version_compare_many
def get_all_compile_args(self) -> T.List[str]:
    """Get the compile arguments from this dependency and it's sub dependencies."""
    return list(itertools.chain(self.get_compile_args(), *(d.get_all_compile_args() for d in self.ext_deps)))