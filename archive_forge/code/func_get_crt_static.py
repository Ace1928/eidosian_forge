from __future__ import annotations
import functools
import subprocess, os.path
import textwrap
import re
import typing as T
from .. import coredata
from ..mesonlib import EnvironmentException, MesonException, Popen_safe_logged, OptionKey
from .compilers import Compiler, clike_debug_args
@functools.lru_cache(maxsize=None)
def get_crt_static(self) -> bool:
    cmd = self.get_exelist(ccache=False) + ['--print', 'cfg']
    p, stdo, stde = Popen_safe_logged(cmd)
    return bool(re.search('^target_feature="crt-static"$', stdo, re.MULTILINE))