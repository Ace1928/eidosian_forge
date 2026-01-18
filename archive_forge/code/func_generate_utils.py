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
def generate_utils(self) -> None:
    self.generate_scanbuild()
    self.generate_clangformat()
    self.generate_clangtidy()
    self.generate_tags('etags', 'TAGS')
    self.generate_tags('ctags', 'ctags')
    self.generate_tags('cscope', 'cscope')
    cmd = self.environment.get_build_command() + ['--internal', 'uninstall']
    elem = self.create_phony_target('uninstall', 'CUSTOM_COMMAND', 'PHONY')
    elem.add_item('COMMAND', cmd)
    elem.add_item('pool', 'console')
    self.add_build(elem)