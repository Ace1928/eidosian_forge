from __future__ import annotations
import re
import os, os.path, pathlib
import shutil
import typing as T
from . import ExtensionModule, ModuleReturnValue, ModuleObject, ModuleInfo
from .. import build, mesonlib, mlog, dependencies
from ..cmake import TargetOptions, cmake_defines_to_args
from ..interpreter import SubprojectHolder
from ..interpreter.type_checking import REQUIRED_KW, INSTALL_DIR_KW, NoneType, in_set_validator
from ..interpreterbase import (
def detect_cmake(self, state: ModuleState) -> bool:
    if self.cmake_detected:
        return True
    cmakebin = state.find_program('cmake', silent=False)
    if not cmakebin.found():
        return False
    p, stdout, stderr = mesonlib.Popen_safe(cmakebin.get_command() + ['--system-information', '-G', 'Ninja'])[0:3]
    if p.returncode != 0:
        mlog.log(f'error retrieving cmake information: returnCode={p.returncode} stdout={stdout} stderr={stderr}')
        return False
    match = re.search('\nCMAKE_ROOT \\"([^"]+)"\n', stdout.strip())
    if not match:
        mlog.log('unable to determine cmake root')
        return False
    cmakePath = pathlib.PurePath(match.group(1))
    self.cmake_root = os.path.join(*cmakePath.parts)
    self.cmake_detected = True
    return True