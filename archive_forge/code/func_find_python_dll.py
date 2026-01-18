import os
import sys
import subprocess
import re
import textwrap
import numpy.distutils.ccompiler  # noqa: F401
from numpy.distutils import log
import distutils.cygwinccompiler
from distutils.unixccompiler import UnixCCompiler
from distutils.msvccompiler import get_build_version as get_build_msvc_version
from distutils.errors import UnknownFileError
from numpy.distutils.misc_util import (msvc_runtime_library,
def find_python_dll():
    stems = [sys.prefix]
    if sys.base_prefix != sys.prefix:
        stems.append(sys.base_prefix)
    sub_dirs = ['', 'lib', 'bin']
    lib_dirs = []
    for stem in stems:
        for folder in sub_dirs:
            lib_dirs.append(os.path.join(stem, folder))
    if 'SYSTEMROOT' in os.environ:
        lib_dirs.append(os.path.join(os.environ['SYSTEMROOT'], 'System32'))
    major_version, minor_version = tuple(sys.version_info[:2])
    implementation = sys.implementation.name
    if implementation == 'cpython':
        dllname = f'python{major_version}{minor_version}.dll'
    elif implementation == 'pypy':
        dllname = f'libpypy{major_version}.{minor_version}-c.dll'
    else:
        dllname = f'Unknown platform {implementation}'
    print('Looking for %s' % dllname)
    for folder in lib_dirs:
        dll = os.path.join(folder, dllname)
        if os.path.exists(dll):
            return dll
    raise ValueError('%s not found in %s' % (dllname, lib_dirs))