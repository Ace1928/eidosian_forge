from __future__ import annotations
import abc
import functools
import os
import multiprocessing
import pathlib
import re
import subprocess
import typing as T
from ... import mesonlib
from ... import mlog
from ...mesonlib import OptionKey
from mesonbuild.compilers.compilers import CompileCheckMode
def gnu_symbol_visibility_args(self, vistype: str) -> T.List[str]:
    if vistype == 'inlineshidden' and self.language not in {'cpp', 'objcpp'}:
        vistype = 'hidden'
    return gnu_symbol_visibility_args[vistype]