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
def replace_paths(self, target, args, override_subdir=None):
    if override_subdir:
        source_target_dir = os.path.join(self.build_to_src, override_subdir)
    else:
        source_target_dir = self.get_target_source_dir(target)
    relout = self.get_target_private_dir(target)
    args = [x.replace('@SOURCE_DIR@', self.build_to_src).replace('@BUILD_DIR@', relout) for x in args]
    args = [x.replace('@CURRENT_SOURCE_DIR@', source_target_dir) for x in args]
    args = [x.replace('@SOURCE_ROOT@', self.build_to_src).replace('@BUILD_ROOT@', '.') for x in args]
    args = [x.replace('\\', '/') for x in args]
    return args