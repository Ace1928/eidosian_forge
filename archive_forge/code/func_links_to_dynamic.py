import os
import sys
import itertools
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.util import cache_from_source as _compiled_file_name
from typing import Dict, Iterator, List, Tuple
from pathlib import Path
from distutils.command.build_ext import build_ext as _du_build_ext
from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler, get_config_var
from distutils import log
from setuptools.errors import BaseError
from setuptools.extension import Extension, Library
from distutils.sysconfig import _config_vars as _CONFIG_VARS  # noqa
def links_to_dynamic(self, ext):
    """Return true if 'ext' links to a dynamic lib in the same package"""
    libnames = dict.fromkeys([lib._full_name for lib in self.shlibs])
    pkg = '.'.join(ext._full_name.split('.')[:-1] + [''])
    return any((pkg + libname in libnames for libname in ext.libraries))