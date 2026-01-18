import abc
import configparser as cp
import fnmatch
from functools import wraps
import inspect
from io import BufferedReader, IOBase
import logging
import os
import os.path as osp
import re
import sys
from git.compat import defenc, force_text
from git.util import LockFile
from typing import (
from git.types import Lit_config_levels, ConfigLevels_Tup, PathLike, assert_never, _T
def rename_section(self, section: str, new_name: str) -> 'GitConfigParser':
    """Rename the given section to new_name.

        :raise ValueError: If ``section`` doesn't exist
        :raise ValueError: If a section with ``new_name`` does already exist
        :return: This instance
        """
    if not self.has_section(section):
        raise ValueError("Source section '%s' doesn't exist" % section)
    if self.has_section(new_name):
        raise ValueError("Destination section '%s' already exists" % new_name)
    super().add_section(new_name)
    new_section = self._sections[new_name]
    for k, vs in self.items_all(section):
        new_section.setall(k, vs)
    self.remove_section(section)
    return self