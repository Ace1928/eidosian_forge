from __future__ import annotations
import itertools
import os, platform, re, sys, shutil
import typing as T
import collections
from . import coredata
from . import mesonlib
from .mesonlib import (
from . import mlog
from .programs import ExternalProgram
from .envconfig import (
from . import compilers
from .compilers import (
from functools import lru_cache
from mesonbuild import envconfig
def detect_kernel(system: str) -> T.Optional[str]:
    if system == 'sunos':
        if mesonlib.version_compare(platform.uname().release, '<=5.10'):
            return 'solaris'
        p, out, _ = Popen_safe(['/usr/bin/uname', '-o'])
        if p.returncode != 0:
            raise MesonException('Failed to run "/usr/bin/uname -o"')
        out = out.lower().strip()
        if out not in {'illumos', 'solaris'}:
            mlog.warning(f'''Got an unexpected value for kernel on a SunOS derived platform, expcted either "illumos" or "solaris", but got "{out}".Please open a Meson issue with the OS you're running and the value detected for your kernel.''')
            return None
        return out
    return KERNEL_MAPPINGS.get(system, None)