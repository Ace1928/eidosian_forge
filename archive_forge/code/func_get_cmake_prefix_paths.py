from __future__ import annotations
import subprocess as S
from threading import Thread
import typing as T
import re
import os
from .. import mlog
from ..mesonlib import PerMachine, Popen_safe, version_compare, is_windows, OptionKey
from ..programs import find_external_program, NonExistingExternalProgram
def get_cmake_prefix_paths(self) -> T.List[str]:
    return self.prefix_paths