from __future__ import annotations
import functools
import os
import typing as T
import subprocess
import re
from .gnu import GnuLikeCompiler
from .gnu import gnu_optimization_args
from ...mesonlib import Popen_safe, OptionKey
def get_option_compile_args(self, options: 'KeyedOptionDictType') -> T.List[str]:
    args: T.List[str] = []
    std = options[OptionKey('std', lang=self.language, machine=self.for_machine)]
    if std.value != 'none':
        args.append('-std=' + std.value)
    return args