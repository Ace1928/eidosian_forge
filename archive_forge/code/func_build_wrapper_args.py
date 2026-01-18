from __future__ import annotations
import collections
import functools
import glob
import itertools
import os
import re
import subprocess
import copy
import typing as T
from pathlib import Path
from ... import arglist
from ... import mesonlib
from ... import mlog
from ...linkers.linkers import GnuLikeDynamicLinkerMixin, SolarisDynamicLinker, CompCertDynamicLinker
from ...mesonlib import LibType, OptionKey
from .. import compilers
from ..compilers import CompileCheckMode
from .visualstudio import VisualStudioLikeCompiler
def build_wrapper_args(self, env: 'Environment', extra_args: T.Union[None, arglist.CompilerArgs, T.List[str], T.Callable[[CompileCheckMode], T.List[str]]], dependencies: T.Optional[T.List['Dependency']], mode: CompileCheckMode=CompileCheckMode.COMPILE) -> arglist.CompilerArgs:
    if extra_args is None:
        extra_args = []
    else:
        extra_args = mesonlib.listify(extra_args)
    extra_args = mesonlib.listify([e(mode.value) if callable(e) else e for e in extra_args])
    if dependencies is None:
        dependencies = []
    elif not isinstance(dependencies, collections.abc.Iterable):
        dependencies = [dependencies]
    cargs: arglist.CompilerArgs = self.compiler_args()
    largs: T.List[str] = []
    for d in dependencies:
        cargs += d.get_compile_args()
        system_incdir = d.get_include_type() == 'system'
        for i in d.get_include_dirs():
            for idir in i.to_string_list(env.get_source_dir(), env.get_build_dir()):
                cargs.extend(self.get_include_args(idir, system_incdir))
        if mode is CompileCheckMode.LINK:
            largs += d.get_link_args()
    ca, la = self._get_basic_compiler_args(env, mode)
    cargs += ca
    largs += la
    cargs += self.get_compiler_check_args(mode)
    if self.linker_to_compiler_args([]) == ['/link'] and largs != [] and ('/link' not in extra_args):
        extra_args += ['/link']
    args = cargs + extra_args + largs
    return args