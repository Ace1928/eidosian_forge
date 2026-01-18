import functools
import importlib
import os
import warnings
from functools import lru_cache
from importlib.util import find_spec
from types import ModuleType
from typing import Any, Callable, List, Optional, TypeVar
import pkg_resources
from packaging.requirements import Requirement
from packaging.version import Version
from typing_extensions import ParamSpec
def get_dependency_min_version_spec(package_name: str, dependency_name: str) -> str:
    """Return the minimum version specifier of a dependency of a package.

    >>> get_dependency_min_version_spec("pytorch-lightning==1.8.0", "jsonargparse")
    '>=4.12.0'

    """
    dependencies = metadata.requires(package_name) or []
    for dep in dependencies:
        dependency = Requirement(dep)
        if dependency.name == dependency_name:
            spec = [str(s) for s in dependency.specifier if str(s)[0] == '>']
            return spec[0] if spec else ''
    raise ValueError(f'This is an internal error. Please file a GitHub issue with the error message. Dependency {dependency_name!r} not found in package {package_name!r}.')