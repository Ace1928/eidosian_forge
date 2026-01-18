from __future__ import annotations
import subprocess as S
from threading import Thread
import typing as T
import re
import os
from .. import mlog
from ..mesonlib import PerMachine, Popen_safe, version_compare, is_windows, OptionKey
from ..programs import find_external_program, NonExistingExternalProgram
def find_cmake_binary(self, environment: 'Environment', silent: bool=False) -> T.Tuple[T.Optional['ExternalProgram'], T.Optional[str]]:
    if isinstance(CMakeExecutor.class_cmakebin[self.for_machine], NonExistingExternalProgram):
        mlog.debug(f'CMake binary for {self.for_machine} is cached as not found')
        return (None, None)
    elif CMakeExecutor.class_cmakebin[self.for_machine] is not None:
        mlog.debug(f'CMake binary for {self.for_machine} is cached.')
    else:
        assert CMakeExecutor.class_cmakebin[self.for_machine] is None
        mlog.debug(f'CMake binary for {self.for_machine} is not cached')
        for potential_cmakebin in find_external_program(environment, self.for_machine, 'cmake', 'CMake', environment.default_cmake, allow_default_for_cross=False):
            version_if_ok = self.check_cmake(potential_cmakebin)
            if not version_if_ok:
                continue
            if not silent:
                mlog.log('Found CMake:', mlog.bold(potential_cmakebin.get_path()), f'({version_if_ok})')
            CMakeExecutor.class_cmakebin[self.for_machine] = potential_cmakebin
            CMakeExecutor.class_cmakevers[self.for_machine] = version_if_ok
            break
        else:
            if not silent:
                mlog.log('Found CMake:', mlog.red('NO'))
            CMakeExecutor.class_cmakebin[self.for_machine] = NonExistingExternalProgram()
            CMakeExecutor.class_cmakevers[self.for_machine] = None
            return (None, None)
    return (CMakeExecutor.class_cmakebin[self.for_machine], CMakeExecutor.class_cmakevers[self.for_machine])