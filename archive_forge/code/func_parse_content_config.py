from __future__ import annotations
import os
import pickle
import typing as t
from .constants import (
from .compat.packaging import (
from .compat.yaml import (
from .io import (
from .util import (
from .data import (
from .config import (
def parse_content_config(data: t.Any) -> ContentConfig:
    """Parse the given dictionary as content config and return it."""
    if not isinstance(data, dict):
        raise Exception('config must be type `dict` not `%s`' % type(data))
    modules = parse_modules_config(data.get('modules', {}))
    python_versions = tuple((version for version in SUPPORTED_PYTHON_VERSIONS if version in CONTROLLER_PYTHON_VERSIONS or version in modules.python_versions))
    py2_support = any((version for version in python_versions if str_to_version(version)[0] == 2))
    return ContentConfig(modules=modules, python_versions=python_versions, py2_support=py2_support)