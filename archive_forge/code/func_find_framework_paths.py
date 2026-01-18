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
def find_framework_paths(self, env: 'Environment') -> T.List[str]:
    """
        These are usually /Library/Frameworks and /System/Library/Frameworks,
        unless you select a particular macOS SDK with the -isysroot flag.
        You can also add to this by setting -F in CFLAGS.
        """
    if self.id != 'clang':
        raise mesonlib.MesonException('Cannot find framework path with non-clang compiler')
    commands = self.get_exelist(ccache=False) + ['-v', '-E', '-']
    commands += self.get_always_args()
    commands += env.coredata.get_external_args(self.for_machine, self.language)
    mlog.debug('Finding framework path by running: ', ' '.join(commands), '\n')
    os_env = os.environ.copy()
    os_env['LC_ALL'] = 'C'
    _, _, stde = mesonlib.Popen_safe(commands, env=os_env, stdin=subprocess.PIPE)
    paths: T.List[str] = []
    for line in stde.split('\n'):
        if '(framework directory)' not in line:
            continue
        paths.append(line[:-21].strip())
    return paths