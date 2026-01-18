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
def should_use_dyndeps_for_target(self, target: 'build.BuildTarget') -> bool:
    if mesonlib.version_compare(self.ninja_version, '<1.10.0'):
        return False
    if 'fortran' in target.compilers:
        return True
    if 'cpp' not in target.compilers:
        return False
    if '-fmodules-ts' in target.extra_args['cpp']:
        return True
    cpp = target.compilers['cpp']
    if cpp.get_id() != 'msvc':
        return False
    cppversion = target.get_option(OptionKey('std', machine=target.for_machine, lang='cpp'))
    if cppversion not in ('latest', 'c++latest', 'vc++latest'):
        return False
    if not mesonlib.current_vs_supports_modules():
        return False
    if mesonlib.version_compare(cpp.version, '<19.28.28617'):
        return False
    return True