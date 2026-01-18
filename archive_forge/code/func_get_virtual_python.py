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
def get_virtual_python(args: EnvironmentConfig, python: VirtualPythonConfig) -> VirtualPythonConfig:
    """Create a virtual environment for the given Python and return the path to its root."""
    if python.system_site_packages:
        suffix = '-ssp'
    else:
        suffix = ''
    virtual_environment_path = os.path.join(ResultType.TMP.path, 'delegation', f'python{python.version}{suffix}')
    virtual_environment_marker = os.path.join(virtual_environment_path, 'marker.txt')
    virtual_environment_python = VirtualPythonConfig(version=python.version, path=os.path.join(virtual_environment_path, 'bin', 'python'), system_site_packages=python.system_site_packages)
    if os.path.exists(virtual_environment_marker):
        display.info('Using existing Python %s virtual environment: %s' % (python.version, virtual_environment_path), verbosity=1)
    else:
        remove_tree(virtual_environment_path)
        if not create_virtual_environment(args, python, virtual_environment_path, python.system_site_packages):
            raise ApplicationError(f'Python {python.version} does not provide virtual environment support.')
        commands = collect_bootstrap(virtual_environment_python)
        run_pip(args, virtual_environment_python, commands, None)
    pathlib.Path(virtual_environment_marker).touch()
    return virtual_environment_python