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
@lru_cache
def get_gitignore(root: Path) -> PathSpec:
    """Return a PathSpec matching gitignore content if present."""
    gitignore = root / '.gitignore'
    lines: List[str] = []
    if gitignore.is_file():
        with gitignore.open(encoding='utf-8') as gf:
            lines = gf.readlines()
    try:
        return PathSpec.from_lines('gitwildmatch', lines)
    except GitWildMatchPatternError as e:
        err(f'Could not parse {gitignore}: {e}')
        raise