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
def machine_info_can_run(machine_info: MachineInfo):
    """Whether we can run binaries for this machine on the current machine.

    Can almost always run 32-bit binaries on 64-bit natively if the host
    and build systems are the same. We don't pass any compilers to
    detect_cpu_family() here because we always want to know the OS
    architecture, not what the compiler environment tells us.
    """
    if machine_info.system != detect_system():
        return False
    true_build_cpu_family = detect_cpu_family({})
    return machine_info.cpu_family == true_build_cpu_family or (true_build_cpu_family == 'x86_64' and machine_info.cpu_family == 'x86') or (true_build_cpu_family == 'mips64' and machine_info.cpu_family == 'mips') or (true_build_cpu_family == 'aarch64' and machine_info.cpu_family == 'arm')