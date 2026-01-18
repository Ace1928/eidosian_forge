from __future__ import annotations
import errno
import os
import site
import stat
import sys
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional
import platformdirs
from .utils import deprecation
def jupyter_runtime_dir() -> str:
    """Return the runtime dir for transient jupyter files.

    Returns JUPYTER_RUNTIME_DIR if defined.

    The default is now (data_dir)/runtime on all platforms;
    we no longer use XDG_RUNTIME_DIR after various problems.
    """
    env = os.environ
    if env.get('JUPYTER_RUNTIME_DIR'):
        return env['JUPYTER_RUNTIME_DIR']
    return pjoin(jupyter_data_dir(), 'runtime')