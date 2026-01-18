from __future__ import annotations
import collections.abc as c
import json
import os
import pathlib
import sys
import typing as t
from .config import (
from .util import (
from .util_common import (
from .host_configs import (
from .python_requirements import (
def create_virtual_environment(args: EnvironmentConfig, python: PythonConfig, path: str, system_site_packages: bool=False, pip: bool=False) -> bool:
    """Create a virtual environment using venv or virtualenv for the requested Python version."""
    if not os.path.exists(python.path):
        return False
    if str_to_version(python.version) >= (3, 0):
        for real_python in iterate_real_pythons(python.version):
            if run_venv(args, real_python, system_site_packages, pip, path):
                display.info('Created Python %s virtual environment using "venv": %s' % (python.version, path), verbosity=1)
                return True
    if run_virtualenv(args, python.path, python.path, system_site_packages, pip, path):
        display.info('Created Python %s virtual environment using "virtualenv": %s' % (python.version, path), verbosity=1)
        return True
    available_pythons = get_available_python_versions()
    for available_python_version, available_python_interpreter in sorted(available_pythons.items()):
        if available_python_interpreter == python.path:
            continue
        virtualenv_version = get_virtualenv_version(args, available_python_interpreter)
        if not virtualenv_version:
            continue
        if run_virtualenv(args, available_python_interpreter, python.path, system_site_packages, pip, path):
            display.info('Created Python %s virtual environment using "virtualenv" on Python %s: %s' % (python.version, available_python_version, path), verbosity=1)
            return True
    return False