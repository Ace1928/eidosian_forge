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
def generate_static_link_rules(self):
    num_pools = self.environment.coredata.options[OptionKey('backend_max_links')].value
    if 'java' in self.environment.coredata.compilers.host:
        self.generate_java_link()
    for for_machine in MachineChoice:
        static_linker = self.build.static_linker[for_machine]
        if static_linker is None:
            continue
        rule = 'STATIC_LINKER{}'.format(self.get_rule_suffix(for_machine))
        cmdlist: T.List[T.Union[str, NinjaCommandArg]] = []
        args = ['$in']
        if isinstance(static_linker, ArLikeLinker) and (not mesonlib.is_windows()):
            cmdlist = execute_wrapper + [c.format('$out') for c in rmfile_prefix]
        cmdlist += static_linker.get_exelist()
        cmdlist += ['$LINK_ARGS']
        cmdlist += NinjaCommandArg.list(static_linker.get_output_args('$out'), Quoting.none)
        if static_linker.id == 'applear':
            cmdlist.extend(args)
            args = []
            ranlib = self.environment.lookup_binary_entry(for_machine, 'ranlib')
            if ranlib is None:
                ranlib = ['ranlib']
            cmdlist.extend(['&&'] + ranlib + ['-c', '$out'])
        description = 'Linking static target $out'
        if num_pools > 0:
            pool = 'pool = link_pool'
        else:
            pool = None
        options = self._rsp_options(static_linker)
        self.add_rule(NinjaRule(rule, cmdlist, args, description, **options, extra=pool))