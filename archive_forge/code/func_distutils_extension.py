from setuptools import distutils as dutils
from setuptools.command import build_ext
from setuptools.extension import Extension
import os
import shutil
import sys
import tempfile
from numba.core import typing, sigutils
from numba.core.compiler_lock import global_compiler_lock
from numba.pycc.compiler import ModuleCompiler, ExportEntry
from numba.pycc.platform import Toolchain
from numba import cext
def distutils_extension(self, **kwargs):
    """
        Create a distutils extension object that can be used in your
        setup.py.
        """
    macros = kwargs.pop('macros', []) + self._get_mixin_defines()
    depends = kwargs.pop('depends', []) + [self._source_path]
    extra_compile_args = kwargs.pop('extra_compile_args', []) + self._get_extra_cflags()
    extra_link_args = kwargs.pop('extra_link_args', []) + self._get_extra_ldflags()
    include_dirs = kwargs.pop('include_dirs', []) + self._toolchain.get_python_include_dirs()
    libraries = kwargs.pop('libraries', []) + self._toolchain.get_python_libraries()
    library_dirs = kwargs.pop('library_dirs', []) + self._toolchain.get_python_library_dirs()
    python_package_path = self._source_module[:self._source_module.rfind('.') + 1]
    ext = _CCExtension(name=python_package_path + self._basename, sources=self._get_mixin_sources(), depends=depends, define_macros=macros, include_dirs=include_dirs, libraries=libraries, library_dirs=library_dirs, extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, **kwargs)
    ext.monkey_patch_distutils()
    ext._cc = self
    return ext