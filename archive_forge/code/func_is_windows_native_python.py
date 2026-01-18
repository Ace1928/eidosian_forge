import os
import pathlib
import platform
import stat
import sys
from logging import getLogger
from typing import Union
def is_windows_native_python() -> bool:
    return sys.platform == 'win32' and os.name == 'nt' and ('cygwin' not in platform.system().lower()) and ('cygwin' not in sys.platform)