import contextlib
import os
import platform
import shlex
import shutil
import sys
import sysconfig
import tempfile
import threading
import warnings
from functools import lru_cache
from pathlib import Path
from typing import (
import distutils.ccompiler
import distutils.errors
def naive_recompile(obj: str, src: str) -> bool:
    """
    This will recompile only if the source file changes. It does not check
    header files, so a more advanced function or Ccache is better if you have
    editable header files in your package.
    """
    return os.stat(obj).st_mtime < os.stat(src).st_mtime