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
def strip_system_includedirs(environment: 'Environment', for_machine: MachineChoice, include_args: T.List[str]) -> T.List[str]:
    """Remove -I<system path> arguments.

    leaving these in will break builds where user want dependencies with system
    include-type used in rust.bindgen targets as if will cause system headers
    to not be found.
    """
    exclude = {f'-I{p}' for p in environment.get_compiler_system_include_dirs(for_machine)}
    return [i for i in include_args if i not in exclude]