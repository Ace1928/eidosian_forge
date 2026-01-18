from __future__ import annotations
import sys, os, subprocess, shutil
import shlex
import typing as T
from .. import envconfig
from .. import mlog
from ..compilers import compilers
from ..compilers.detect import defaults as compiler_names
def detect_cross_system(infos: MachineInfo, options: T.Any) -> None:
    for optname in ('system', 'subsystem', 'kernel', 'cpu', 'cpu_family', 'endian'):
        v = getattr(options, optname)
        if not v:
            mlog.error(f'Cross property "{optname}" missing, set it with --{optname.replace('_', '-')}.')
            sys.exit(1)
        setattr(infos, optname, v)