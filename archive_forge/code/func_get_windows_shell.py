from __future__ import annotations
import os, subprocess
import argparse
import tempfile
import shutil
import itertools
import typing as T
from pathlib import Path
from . import build, minstall
from .mesonlib import (EnvironmentVariables, MesonException, is_windows, setup_vsenv, OptionKey,
from . import mlog
def get_windows_shell() -> T.Optional[str]:
    mesonbuild = Path(__file__).parent
    script = mesonbuild / 'scripts' / 'cmd_or_ps.ps1'
    for shell in POWERSHELL_EXES:
        try:
            command = [shell, '-noprofile', '-executionpolicy', 'bypass', '-file', str(script)]
            result = subprocess.check_output(command)
            return result.decode().strip()
        except (subprocess.CalledProcessError, OSError):
            pass
    return None