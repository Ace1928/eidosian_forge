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
def setup_shlib_compiler(self):
    compiler = self.shlib_compiler = new_compiler(compiler=self.compiler, dry_run=self.dry_run, force=self.force)
    _customize_compiler_for_shlib(compiler)
    if self.include_dirs is not None:
        compiler.set_include_dirs(self.include_dirs)
    if self.define is not None:
        for name, value in self.define:
            compiler.define_macro(name, value)
    if self.undef is not None:
        for macro in self.undef:
            compiler.undefine_macro(macro)
    if self.libraries is not None:
        compiler.set_libraries(self.libraries)
    if self.library_dirs is not None:
        compiler.set_library_dirs(self.library_dirs)
    if self.rpath is not None:
        compiler.set_runtime_library_dirs(self.rpath)
    if self.link_objects is not None:
        compiler.set_link_objects(self.link_objects)
    compiler.link_shared_object = link_shared_object.__get__(compiler)