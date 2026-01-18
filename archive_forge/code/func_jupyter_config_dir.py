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
def jupyter_config_dir() -> str:
    """Get the Jupyter config directory for this platform and user.

    Returns JUPYTER_CONFIG_DIR if defined, otherwise the appropriate
    directory for the platform.
    """
    env = os.environ
    if env.get('JUPYTER_NO_CONFIG'):
        return _mkdtemp_once('jupyter-clean-cfg')
    if env.get('JUPYTER_CONFIG_DIR'):
        return env['JUPYTER_CONFIG_DIR']
    if use_platform_dirs():
        return platformdirs.user_config_dir(APPNAME, appauthor=False)
    home_dir = get_home_dir()
    return pjoin(home_dir, '.jupyter')