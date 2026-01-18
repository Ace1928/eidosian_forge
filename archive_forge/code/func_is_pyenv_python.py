import ast
import logging
import os
import re
import sys
import warnings
from typing import List
from importlib import util
from importlib.metadata import version
from pathlib import Path
from . import Nuitka, run_command
def is_pyenv_python(self):
    pyenv_root = os.environ.get('PYENV_ROOT')
    if pyenv_root:
        resolved_exe = self.exe.resolve()
        if str(resolved_exe).startswith(pyenv_root):
            return True
    return False