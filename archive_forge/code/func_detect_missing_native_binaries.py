from __future__ import annotations
import sys, os, subprocess, shutil
import shlex
import typing as T
from .. import envconfig
from .. import mlog
from ..compilers import compilers
from ..compilers.detect import defaults as compiler_names
def detect_missing_native_binaries(infos: MachineInfo) -> None:
    for toolname in sorted(envconfig.ENV_VAR_TOOL_MAP.keys()):
        if toolname in infos.binaries:
            continue
        exe = shutil.which(toolname)
        if exe:
            infos.binaries[toolname] = [exe]