from __future__ import annotations
import subprocess as S
from threading import Thread
import typing as T
import re
import os
from .. import mlog
from ..mesonlib import PerMachine, Popen_safe, version_compare, is_windows, OptionKey
from ..programs import find_external_program, NonExistingExternalProgram
def check_cmake(self, cmakebin: 'ExternalProgram') -> T.Optional[str]:
    if not cmakebin.found():
        mlog.log(f'Did not find CMake {cmakebin.name!r}')
        return None
    try:
        cmd = cmakebin.get_command()
        p, out = Popen_safe(cmd + ['--version'])[0:2]
        if p.returncode != 0:
            mlog.warning("Found CMake {!r} but couldn't run it".format(' '.join(cmd)))
            return None
    except FileNotFoundError:
        mlog.warning("We thought we found CMake {!r} but now it's not there. How odd!".format(' '.join(cmd)))
        return None
    except PermissionError:
        msg = "Found CMake {!r} but didn't have permissions to run it.".format(' '.join(cmd))
        if not is_windows():
            msg += '\n\nOn Unix-like systems this is often caused by scripts that are not executable.'
        mlog.warning(msg)
        return None
    cmvers = re.search('(cmake|cmake3)\\s*version\\s*([\\d.]+)', out)
    if cmvers is not None:
        return cmvers.group(2)
    mlog.warning(f'We thought we found CMake {cmd!r}, but it was missing the expected version string in its output.')
    return None