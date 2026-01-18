from __future__ import annotations
from pathlib import Path
from .base import ExternalDependency, DependencyException, sort_libpaths, DependencyTypeName
from ..mesonlib import EnvironmentVariables, OptionKey, OrderedSet, PerMachine, Popen_safe, Popen_safe_logged, MachineChoice, join_args
from ..programs import find_external_program, ExternalProgram
from .. import mlog
from pathlib import PurePath
from functools import lru_cache
import re
import os
import shlex
import typing as T
def extract_libtool_shlib(self, la_file: str) -> T.Optional[str]:
    """
        Returns the path to the shared library
        corresponding to this .la file
        """
    dlname = self.extract_dlname_field(la_file)
    if dlname is None:
        return None
    if self.env.machines[self.for_machine].is_darwin():
        dlbasename = os.path.basename(dlname)
        libdir = self.extract_libdir_field(la_file)
        if libdir is None:
            return dlbasename
        return os.path.join(libdir, dlbasename)
    return os.path.basename(dlname)