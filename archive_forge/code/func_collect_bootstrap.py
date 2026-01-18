from __future__ import annotations
import base64
import dataclasses
import json
import os
import re
import typing as t
from .encoding import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .data import (
from .host_configs import (
from .connections import (
from .coverage_util import (
def collect_bootstrap(python: PythonConfig) -> list[PipCommand]:
    """Return the details necessary to bootstrap pip into an empty virtual environment."""
    infrastructure_packages = get_venv_packages(python)
    pip_version = infrastructure_packages['pip']
    packages = [f'{name}=={version}' for name, version in infrastructure_packages.items()]
    bootstrap = PipBootstrap(pip_version=pip_version, packages=packages)
    return [bootstrap]