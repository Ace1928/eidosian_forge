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
def create_phony_target(self, dummy_outfile: str, rulename: str, phony_infilename: str) -> NinjaBuildElement:
    """
        We need to use aliases for targets that might be used as directory
        names to workaround a Ninja bug that breaks `ninja -t clean`.
        This is used for 'reserved' targets such as 'test', 'install',
        'benchmark', etc, and also for RunTargets.
        https://github.com/mesonbuild/meson/issues/1644
        """
    if dummy_outfile.startswith('meson-internal__'):
        raise AssertionError(f'Invalid usage of create_phony_target with {dummy_outfile!r}')
    to_name = f'meson-internal__{dummy_outfile}'
    elem = NinjaBuildElement(self.all_outputs, dummy_outfile, 'phony', to_name)
    self.add_build(elem)
    return NinjaBuildElement(self.all_outputs, to_name, rulename, phony_infilename)