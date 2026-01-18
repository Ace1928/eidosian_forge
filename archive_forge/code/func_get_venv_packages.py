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
def get_venv_packages(python: PythonConfig) -> dict[str, str]:
    """Return a dictionary of Python packages needed for a consistent virtual environment specific to the given Python version."""
    default_packages = dict(pip='23.1.2', setuptools='67.7.2', wheel='0.37.1')
    override_packages = {'2.7': dict(pip='20.3.4', setuptools='44.1.1', wheel=None), '3.6': dict(pip='21.3.1', setuptools='59.6.0', wheel=None)}
    packages = {name: version or default_packages[name] for name, version in override_packages.get(python.version, default_packages).items()}
    return packages