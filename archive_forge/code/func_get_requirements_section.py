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
def get_requirements_section(launch_project: LaunchProject, builder_type: str) -> str:
    if builder_type == 'docker':
        buildx_installed = docker.is_buildx_installed()
        if not buildx_installed:
            wandb.termwarn('Docker BuildX is not installed, for faster builds upgrade docker: https://github.com/docker/buildx#installing')
            prefix = 'RUN WANDB_DISABLE_CACHE=true'
    elif builder_type == 'kaniko':
        prefix = 'RUN WANDB_DISABLE_CACHE=true'
        buildx_installed = False
    if launch_project.deps_type == 'pip':
        requirements_files = []
        deps_install_line = None
        assert launch_project.project_dir is not None
        base_path = pathlib.Path(launch_project.project_dir)
        if (base_path / 'requirements.txt').exists():
            requirements_files += ['src/requirements.txt']
            deps_install_line = 'pip install -r requirements.txt'
        elif (base_path / 'pyproject.toml').exists():
            tomli = get_module('tomli')
            if tomli is None:
                wandb.termwarn('pyproject.toml found but tomli could not be loaded. To install dependencies from pyproject.toml please run `pip install tomli` and try again.')
            else:
                with open(base_path / 'pyproject.toml', 'rb') as f:
                    contents = tomli.load(f)
                project_deps = [str(d) for d in contents.get('project', {}).get('dependencies', [])]
                if project_deps:
                    with open(base_path / 'requirements.txt', 'w') as f:
                        f.write('\n'.join(project_deps))
                    requirements_files += ['src/requirements.txt']
                    deps_install_line = 'pip install -r requirements.txt'
        if not deps_install_line and (base_path / 'requirements.frozen.txt').exists():
            requirements_files += ['src/requirements.frozen.txt', '_wandb_bootstrap.py']
            deps_install_line = _parse_existing_requirements(launch_project) + 'python _wandb_bootstrap.py'
        if not deps_install_line:
            raise LaunchError(f'No dependency sources found for {launch_project}')
        if buildx_installed:
            prefix = 'RUN --mount=type=cache,mode=0777,target=/root/.cache/pip'
        requirements_line = PIP_TEMPLATE.format(buildx_optional_prefix=prefix, requirements_files=' '.join(requirements_files), pip_install=deps_install_line)
    elif launch_project.deps_type == 'conda':
        if buildx_installed:
            prefix = 'RUN --mount=type=cache,mode=0777,target=/opt/conda/pkgs'
        requirements_line = CONDA_TEMPLATE.format(buildx_optional_prefix=prefix)
    else:
        requirements_line = 'RUN mkdir -p env/'
        wandb.termwarn('No requirements file found. No packages will be installed.')
    return requirements_line