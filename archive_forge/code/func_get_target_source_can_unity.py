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
def get_target_source_can_unity(self, target, source):
    if isinstance(source, File):
        source = source.fname
    if self.environment.is_llvm_ir(source) or self.environment.is_assembly(source):
        return False
    suffix = os.path.splitext(source)[1][1:].lower()
    for lang in backends.LANGS_CANT_UNITY:
        if lang not in target.compilers:
            continue
        if suffix in target.compilers[lang].file_suffixes:
            return False
    return True