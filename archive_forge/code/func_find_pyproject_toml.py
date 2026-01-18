import io
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import (
from mypy_extensions import mypyc_attr
from packaging.specifiers import InvalidSpecifier, Specifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from pathspec import PathSpec
from pathspec.patterns.gitwildmatch import GitWildMatchPatternError
from black.handle_ipynb_magics import jupyter_dependencies_are_installed
from black.mode import TargetVersion
from black.output import err
from black.report import Report
def find_pyproject_toml(path_search_start: Tuple[str, ...], stdin_filename: Optional[str]=None) -> Optional[str]:
    """Find the absolute filepath to a pyproject.toml if it exists"""
    path_project_root, _ = find_project_root(path_search_start, stdin_filename)
    path_pyproject_toml = path_project_root / 'pyproject.toml'
    if path_pyproject_toml.is_file():
        return str(path_pyproject_toml)
    try:
        path_user_pyproject_toml = find_user_pyproject_toml()
        return str(path_user_pyproject_toml) if path_user_pyproject_toml.is_file() else None
    except (PermissionError, RuntimeError) as e:
        err(f'Ignoring user configuration directory due to {e!r}')
        return None