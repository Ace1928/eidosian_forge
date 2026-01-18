from distutils.dir_util import remove_tree, mkpath
from distutils import log
from types import CodeType
import sys
import os
import re
import textwrap
import marshal
from setuptools.extension import Library
from setuptools import Command
from .._path import ensure_directory
from sysconfig import get_path, get_python_version
def do_install_data(self):
    self.get_finalized_command('install').install_lib = self.bdist_dir
    site_packages = os.path.normcase(os.path.realpath(_get_purelib()))
    old, self.distribution.data_files = (self.distribution.data_files, [])
    for item in old:
        if isinstance(item, tuple) and len(item) == 2:
            if os.path.isabs(item[0]):
                realpath = os.path.realpath(item[0])
                normalized = os.path.normcase(realpath)
                if normalized == site_packages or normalized.startswith(site_packages + os.sep):
                    item = (realpath[len(site_packages) + 1:], item[1])
        self.distribution.data_files.append(item)
    try:
        log.info('installing package data to %s', self.bdist_dir)
        self.call_command('install_data', force=0, root=None)
    finally:
        self.distribution.data_files = old