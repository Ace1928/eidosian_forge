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
def get_abi3_suffix():
    """Return the file extension for an abi3-compliant Extension()"""
    for suffix in EXTENSION_SUFFIXES:
        if '.abi3' in suffix:
            return suffix
        elif suffix == '.pyd':
            return suffix
    return None