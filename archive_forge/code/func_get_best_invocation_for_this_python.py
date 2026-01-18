import itertools
import os
import shutil
import sys
from typing import List, Optional
from pip._internal.cli.main import main
from pip._internal.utils.compat import WINDOWS
def get_best_invocation_for_this_python() -> str:
    """Try to figure out the best way to invoke the current Python."""
    exe = sys.executable
    exe_name = os.path.basename(exe)
    found_executable = shutil.which(exe_name)
    if found_executable and os.path.samefile(found_executable, exe):
        return exe_name
    return exe