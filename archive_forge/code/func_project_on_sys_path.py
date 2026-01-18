import os
import operator
import sys
import contextlib
import itertools
import unittest
from distutils.errors import DistutilsError, DistutilsOptionError
from distutils import log
from unittest import TestLoader
from pkg_resources import (
from .._importlib import metadata
from setuptools import Command
from setuptools.extern.more_itertools import unique_everseen
from setuptools.extern.jaraco.functools import pass_none
@contextlib.contextmanager
def project_on_sys_path(self, include_dists=[]):
    self.run_command('egg_info')
    self.reinitialize_command('build_ext', inplace=1)
    self.run_command('build_ext')
    ei_cmd = self.get_finalized_command('egg_info')
    old_path = sys.path[:]
    old_modules = sys.modules.copy()
    try:
        project_path = normalize_path(ei_cmd.egg_base)
        sys.path.insert(0, project_path)
        working_set.__init__()
        add_activation_listener(lambda dist: dist.activate())
        require('%s==%s' % (ei_cmd.egg_name, ei_cmd.egg_version))
        with self.paths_on_pythonpath([project_path]):
            yield
    finally:
        sys.path[:] = old_path
        sys.modules.clear()
        sys.modules.update(old_modules)
        working_set.__init__()