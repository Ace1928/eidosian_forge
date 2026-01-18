import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
def analyse_name(self):
    """The packages/modules are the essential contribution of the author.
        Therefore the name of the distribution can be derived from them.
        """
    if self.dist.metadata.name or self.dist.name:
        return
    log.debug('No `name` configuration, performing automatic discovery')
    name = self._find_name_single_package_or_module() or self._find_name_from_packages()
    if name:
        self.dist.metadata.name = name