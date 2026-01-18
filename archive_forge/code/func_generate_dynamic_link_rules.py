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
def generate_dynamic_link_rules(self):
    num_pools = self.environment.coredata.options[OptionKey('backend_max_links')].value
    for for_machine in MachineChoice:
        complist = self.environment.coredata.compilers[for_machine]
        for langname, compiler in complist.items():
            if langname in {'java', 'vala', 'rust', 'cs', 'cython'}:
                continue
            rule = '{}_LINKER{}'.format(langname, self.get_rule_suffix(for_machine))
            command = compiler.get_linker_exelist()
            args = ['$ARGS'] + NinjaCommandArg.list(compiler.get_linker_output_args('$out'), Quoting.none) + ['$in', '$LINK_ARGS']
            description = 'Linking target $out'
            if num_pools > 0:
                pool = 'pool = link_pool'
            else:
                pool = None
            options = self._rsp_options(compiler)
            self.add_rule(NinjaRule(rule, command, args, description, **options, extra=pool))
        if self.environment.machines[for_machine].is_aix():
            rule = 'AIX_LINKER{}'.format(self.get_rule_suffix(for_machine))
            description = 'Archiving AIX shared library'
            cmdlist = compiler.get_command_to_archive_shlib()
            args = []
            options = {}
            self.add_rule(NinjaRule(rule, cmdlist, args, description, **options, extra=None))
    args = self.environment.get_build_command() + ['--internal', 'symbolextractor', self.environment.get_build_dir(), '$in', '$IMPLIB', '$out']
    symrule = 'SHSYM'
    symcmd = args + ['$CROSS']
    syndesc = 'Generating symbol file $out'
    synstat = 'restat = 1'
    self.add_rule(NinjaRule(symrule, symcmd, [], syndesc, extra=synstat))