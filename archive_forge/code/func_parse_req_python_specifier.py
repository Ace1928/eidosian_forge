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
def parse_req_python_specifier(requires_python: str) -> Optional[List[TargetVersion]]:
    """Parse a specifier string (i.e. ``">=3.7,<3.10"``) to a list of TargetVersion.

    If parsing fails, will raise a packaging.specifiers.InvalidSpecifier error.
    If the parsed specifier cannot be mapped to a valid TargetVersion, returns None.
    """
    specifier_set = strip_specifier_set(SpecifierSet(requires_python))
    if not specifier_set:
        return None
    target_version_map = {f'3.{v.value}': v for v in TargetVersion}
    compatible_versions: List[str] = list(specifier_set.filter(target_version_map))
    if compatible_versions:
        return [target_version_map[v] for v in compatible_versions]
    return None