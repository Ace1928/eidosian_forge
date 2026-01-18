import hashlib
import json
import logging
import os
import pathlib
import shlex
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import yaml
from dockerpycreds.utils import find_executable  # type: ignore
from six.moves import shlex_quote
import wandb
import wandb.docker as docker
import wandb.env
from wandb.apis.internal import Api
from wandb.sdk.launch.loader import (
from wandb.util import get_module
from .._project_spec import EntryPoint, EntrypointDefaults, LaunchProject
from ..errors import ExecutionError, LaunchError
from ..registry.abstract import AbstractRegistry
from ..registry.anon import AnonynmousRegistry
from ..utils import (
def get_base_setup(launch_project: LaunchProject, py_version: str, py_major: str) -> str:
    """Fill in the Dockerfile templates for stage 2 of build.

    CPU version is built on python, Accelerator version is built on user provided.
    """
    python_base_image = f'python:{py_version}-buster'
    if launch_project.accelerator_base_image:
        _logger.info(f'Using accelerator base image: {launch_project.accelerator_base_image}')
        if py_major == '2':
            python_packages = [f'python{py_version}', f'libpython{py_version}', 'python-pip', 'python-setuptools']
        else:
            python_packages = [f'python{py_version}', f'libpython{py_version}', 'python3-pip', 'python3-setuptools']
        base_setup = ACCELERATOR_SETUP_TEMPLATE.format(accelerator_base_image=launch_project.accelerator_base_image, python_packages=' \\\n'.join(python_packages), py_version=py_version)
    else:
        python_packages = ['python3-dev' if py_major == '3' else 'python-dev', 'gcc']
        base_setup = PYTHON_SETUP_TEMPLATE.format(py_base_image=python_base_image)
    return base_setup