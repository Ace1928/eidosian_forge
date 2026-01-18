from __future__ import annotations
import sys, os, subprocess, shutil
import shlex
import typing as T
from .. import envconfig
from .. import mlog
from ..compilers import compilers
from ..compilers.detect import defaults as compiler_names
def has_for_build() -> bool:
    for cenv in envconfig.ENV_VAR_COMPILER_MAP.values():
        if os.environ.get(cenv + '_FOR_BUILD'):
            return True
    return False