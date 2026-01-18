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
def cross_alignment(self, typename: str, prefix: str, env: 'Environment', *, extra_args: T.Optional[T.List[str]]=None, dependencies: T.Optional[T.List['Dependency']]=None) -> int:
    if extra_args is None:
        extra_args = []
    t = f'{prefix}\n        #include <stddef.h>\n        int main(void) {{\n            {typename} something;\n            return 0;\n        }}'
    if not self.compiles(t, env, extra_args=extra_args, dependencies=dependencies)[0]:
        return -1
    t = f'{prefix}\n        #include <stddef.h>\n        struct tmp {{\n            char c;\n            {typename} target;\n        }};'
    return self.cross_compute_int('offsetof(struct tmp, target)', None, None, None, t, env, extra_args, dependencies)