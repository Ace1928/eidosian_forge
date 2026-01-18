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
def generate_single_java_compile(self, src, target, compiler, args):
    deps = [os.path.join(self.get_target_dir(l), l.get_filename()) for l in target.link_targets]
    generated_sources = self.get_target_generated_sources(target)
    for rel_src in generated_sources.keys():
        if rel_src.endswith('.java'):
            deps.append(rel_src)
    rel_src = src.rel_to_builddir(self.build_to_src)
    plain_class_path = src.fname[:-4] + 'class'
    rel_obj = os.path.join(self.get_target_private_dir(target), plain_class_path)
    element = NinjaBuildElement(self.all_outputs, rel_obj, self.compiler_to_rule_name(compiler), rel_src)
    element.add_dep(deps)
    element.add_item('ARGS', args)
    self.add_build(element)
    return plain_class_path