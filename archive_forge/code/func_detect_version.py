from __future__ import annotations
import os
import re
import subprocess
import typing as T
from .. import mlog
from .. import mesonlib
from ..compilers.compilers import CrossNoRunException
from ..mesonlib import (
from ..environment import detect_cpu_family
from .base import DependencyException, DependencyMethods, DependencyTypeName, SystemDependency
from .configtool import ConfigToolDependency
from .detect import packages
from .factory import DependencyFactory
def detect_version(self) -> str:
    gmake = self.get_config_value(['--variable=GNUMAKE'], 'variable')[0]
    makefile_dir = self.get_config_value(['--variable=GNUSTEP_MAKEFILES'], 'variable')[0]
    base_make = os.path.join(makefile_dir, 'Additional', 'base.make')
    printver = "print-%:\n\t@echo '$($*)'"
    env = os.environ.copy()
    env['FOUNDATION_LIB'] = 'gnu'
    p, o, e = Popen_safe([gmake, '-f', '-', '-f', base_make, 'print-GNUSTEP_BASE_VERSION'], env=env, write=printver, stdin=subprocess.PIPE)
    version = o.strip()
    if not version:
        mlog.debug("Couldn't detect GNUStep version, falling back to '1'")
        version = '1'
    return version