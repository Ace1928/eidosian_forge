import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
def construct_package_dir(packages: List[str], package_path: _Path) -> Dict[str, str]:
    parent_pkgs = remove_nested_packages(packages)
    prefix = Path(package_path).parts
    return {pkg: '/'.join([*prefix, *pkg.split('.')]) for pkg in parent_pkgs}