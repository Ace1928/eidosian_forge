from __future__ import annotations
import functools, json, os, textwrap
from pathlib import Path
import typing as T
from .. import mesonlib, mlog
from .base import process_method_kw, DependencyException, DependencyMethods, DependencyTypeName, ExternalDependency, SystemDependency
from .configtool import ConfigToolDependency
from .detect import packages
from .factory import DependencyFactory
from .framework import ExtraFrameworkDependency
from .pkgconfig import PkgConfigDependency
from ..environment import detect_cpu_family
from ..programs import ExternalProgram
def get_windows_link_args(self, limited_api: bool) -> T.Optional[T.List[str]]:
    if self.platform.startswith('win'):
        vernum = self.variables.get('py_version_nodot')
        verdot = self.variables.get('py_version_short')
        imp_lower = self.variables.get('implementation_lower', 'python')
        if self.static:
            libpath = Path('libs') / f'libpython{vernum}.a'
        else:
            comp = self.get_compiler()
            if comp.id == 'gcc':
                if imp_lower == 'pypy' and verdot == '3.8':
                    libpath = Path('libpypy3-c.dll')
                elif imp_lower == 'pypy':
                    libpath = Path(f'libpypy{verdot}-c.dll')
                else:
                    libpath = Path(f'python{vernum}.dll')
            else:
                if limited_api:
                    vernum = vernum[0]
                libpath = Path('libs') / f'python{vernum}.lib'
                buildtype = self.env.coredata.get_option(mesonlib.OptionKey('buildtype'))
                assert isinstance(buildtype, str)
                debug = self.env.coredata.get_option(mesonlib.OptionKey('debug'))
                is_debug_build = debug or buildtype == 'debug'
                vscrt_debug = False
                if mesonlib.OptionKey('b_vscrt') in self.env.coredata.options:
                    vscrt = self.env.coredata.options[mesonlib.OptionKey('b_vscrt')].value
                    if vscrt in {'mdd', 'mtd', 'from_buildtype', 'static_from_buildtype'}:
                        vscrt_debug = True
                if is_debug_build and vscrt_debug and (not self.variables.get('Py_DEBUG')):
                    mlog.warning(textwrap.dedent('                            Using a debug build type with MSVC or an MSVC-compatible compiler\n                            when the Python interpreter is not also a debug build will almost\n                            certainly result in a failed build. Prefer using a release build\n                            type or a debug Python interpreter.\n                            '))
        lib = Path(self.variables.get('base_prefix')) / libpath
    elif self.platform.startswith('mingw'):
        if self.static:
            libname = self.variables.get('LIBRARY')
        else:
            libname = self.variables.get('LDLIBRARY')
        lib = Path(self.variables.get('LIBDIR')) / libname
    else:
        raise mesonlib.MesonBugException("On a Windows path, but the OS doesn't appear to be Windows or MinGW.")
    if not lib.exists():
        mlog.log('Could not find Python3 library {!r}'.format(str(lib)))
        return None
    return [str(lib)]