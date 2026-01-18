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
def parse_req_python_version(requires_python: str) -> Optional[List[TargetVersion]]:
    """Parse a version string (i.e. ``"3.7"``) to a list of TargetVersion.

    If parsing fails, will raise a packaging.version.InvalidVersion error.
    If the parsed version cannot be mapped to a valid TargetVersion, returns None.
    """
    version = Version(requires_python)
    if version.release[0] != 3:
        return None
    try:
        return [TargetVersion(version.release[1])]
    except (IndexError, ValueError):
        return None