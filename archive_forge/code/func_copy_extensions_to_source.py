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
def copy_extensions_to_source(self):
    build_py = self.get_finalized_command('build_py')
    for ext in self.extensions:
        inplace_file, regular_file = self._get_inplace_equivalent(build_py, ext)
        if os.path.exists(regular_file) or not ext.optional:
            self.copy_file(regular_file, inplace_file, level=self.verbose)
        if ext._needs_stub:
            inplace_stub = self._get_equivalent_stub(ext, inplace_file)
            self._write_stub_file(inplace_stub, ext, compile=True)