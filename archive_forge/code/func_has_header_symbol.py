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
def has_header_symbol(self, hname: str, symbol: str, prefix: str, env: 'Environment', *, extra_args: T.Union[None, T.List[str], T.Callable[[CompileCheckMode], T.List[str]]]=None, dependencies: T.Optional[T.List['Dependency']]=None) -> T.Tuple[bool, bool]:
    t = f"{prefix}\n        #include <{hname}>\n        int main(void) {{\n            /* If it's not defined as a macro, try to use as a symbol */\n            #ifndef {symbol}\n                {symbol};\n            #endif\n            return 0;\n        }}"
    return self.compiles(t, env, extra_args=extra_args, dependencies=dependencies)