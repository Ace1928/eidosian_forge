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
def get_default_include_dirs(self) -> T.List[str]:
    return gnulike_default_include_dirs(tuple(self.get_exelist(ccache=False)), self.language).copy()