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
def infer_target_version(pyproject_toml: Dict[str, Any]) -> Optional[List[TargetVersion]]:
    """Infer Black's target version from the project metadata in pyproject.toml.

    Supports the PyPA standard format (PEP 621):
    https://packaging.python.org/en/latest/specifications/declaring-project-metadata/#requires-python

    If the target version cannot be inferred, returns None.
    """
    project_metadata = pyproject_toml.get('project', {})
    requires_python = project_metadata.get('requires-python', None)
    if requires_python is not None:
        try:
            return parse_req_python_version(requires_python)
        except InvalidVersion:
            pass
        try:
            return parse_req_python_specifier(requires_python)
        except (InvalidSpecifier, InvalidVersion):
            pass
    return None