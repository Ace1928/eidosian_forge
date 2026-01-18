from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, unique
from functools import lru_cache
from pathlib import PurePath, Path
from textwrap import dedent
import itertools
import json
import os
import pickle
import re
import subprocess
import typing as T
from . import backends
from .. import modules
from .. import environment, mesonlib
from .. import build
from .. import mlog
from .. import compilers
from ..arglist import CompilerArgs
from ..compilers import Compiler
from ..linkers import ArLikeLinker, RSPFileSyntax
from ..mesonlib import (
from ..mesonlib import get_compiler_for_source, has_path_sep, OptionKey
from .backends import CleanTrees
from ..build import GeneratedList, InvalidArguments
def scan_fortran_module_outputs(self, target):
    """
        Find all module and submodule made available in a Fortran code file.
        """
    if self.use_dyndeps_for_fortran():
        return
    compiler = None
    for lang, c in self.environment.coredata.compilers.host.items():
        if lang == 'fortran':
            compiler = c
            break
    if compiler is None:
        self.fortran_deps[target.get_basename()] = {}
        return
    modre = re.compile(FORTRAN_MODULE_PAT, re.IGNORECASE)
    submodre = re.compile(FORTRAN_SUBMOD_PAT, re.IGNORECASE)
    module_files = {}
    submodule_files = {}
    for s in target.get_sources():
        if not compiler.can_compile(s):
            continue
        filename = s.absolute_path(self.environment.get_source_dir(), self.environment.get_build_dir())
        with open(filename, encoding='ascii', errors='ignore') as f:
            for line in f:
                modmatch = modre.match(line)
                if modmatch is not None:
                    modname = modmatch.group(1).lower()
                    if modname in module_files:
                        raise InvalidArguments(f'Namespace collision: module {modname} defined in two files {module_files[modname]} and {s}.')
                    module_files[modname] = s
                else:
                    submodmatch = submodre.match(line)
                    if submodmatch is not None:
                        parents = submodmatch.group(1).lower().split(':')
                        submodname = parents[0] + '_' + submodmatch.group(2).lower()
                        if submodname in submodule_files:
                            raise InvalidArguments(f'Namespace collision: submodule {submodname} defined in two files {submodule_files[submodname]} and {s}.')
                        submodule_files[submodname] = s
    self.fortran_deps[target.get_basename()] = {**module_files, **submodule_files}