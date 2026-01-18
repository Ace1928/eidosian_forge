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
def get_windows_python_arch(self) -> str:
    if self.platform.startswith('mingw'):
        if 'x86_64' in self.platform:
            return 'x86_64'
        elif 'i686' in self.platform:
            return 'x86'
        elif 'aarch64' in self.platform:
            return 'aarch64'
        else:
            raise DependencyException(f'MinGW Python built with unknown platform {self.platform!r}, please file a bug')
    elif self.platform == 'win32':
        return 'x86'
    elif self.platform in {'win64', 'win-amd64'}:
        return 'x86_64'
    elif self.platform in {'win-arm64'}:
        return 'aarch64'
    raise DependencyException('Unknown Windows Python platform {self.platform!r}')