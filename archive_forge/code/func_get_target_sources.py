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
def get_target_sources(self, target: build.BuildTarget) -> T.MutableMapping[str, File]:
    srcs: T.MutableMapping[str, File] = OrderedDict()
    for s in target.get_sources():
        if not isinstance(s, File):
            raise InvalidArguments(f'All sources in target {s!r} must be of type mesonlib.File')
        f = s.rel_to_builddir(self.build_to_src)
        srcs[f] = s
    return srcs