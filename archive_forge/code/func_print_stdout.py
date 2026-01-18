from __future__ import annotations
import subprocess as S
from threading import Thread
import typing as T
import re
import os
from .. import mlog
from ..mesonlib import PerMachine, Popen_safe, version_compare, is_windows, OptionKey
from ..programs import find_external_program, NonExistingExternalProgram
def print_stdout() -> None:
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        mlog.log(line.decode(errors='ignore').strip('\n'))
    proc.stdout.close()