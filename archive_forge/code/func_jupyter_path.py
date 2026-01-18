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
def jupyter_path(*subdirs: str) -> list[str]:
    """Return a list of directories to search for data files

    JUPYTER_PATH environment variable has highest priority.

    If the JUPYTER_PREFER_ENV_PATH environment variable is set, the environment-level
    directories will have priority over user-level directories.

    If the Python site.ENABLE_USER_SITE variable is True, we also add the
    appropriate Python user site subdirectory to the user-level directories.


    If ``*subdirs`` are given, that subdirectory will be added to each element.

    Examples:

    >>> jupyter_path()
    ['~/.local/jupyter', '/usr/local/share/jupyter']
    >>> jupyter_path('kernels')
    ['~/.local/jupyter/kernels', '/usr/local/share/jupyter/kernels']
    """
    paths: list[str] = []
    if os.environ.get('JUPYTER_PATH'):
        paths.extend((p.rstrip(os.sep) for p in os.environ['JUPYTER_PATH'].split(os.pathsep)))
    user = [jupyter_data_dir()]
    if site.ENABLE_USER_SITE:
        userbase: Optional[str]
        userbase = site.getuserbase() if hasattr(site, 'getuserbase') else site.USER_BASE
        if userbase:
            userdir = str(Path(userbase, 'share', 'jupyter'))
            if userdir not in user:
                user.append(userdir)
    env = [p for p in ENV_JUPYTER_PATH if p not in SYSTEM_JUPYTER_PATH]
    if prefer_environment_over_user():
        paths.extend(env)
        paths.extend(user)
    else:
        paths.extend(user)
        paths.extend(env)
    paths.extend(SYSTEM_JUPYTER_PATH)
    if subdirs:
        paths = [pjoin(p, *subdirs) for p in paths]
    return paths