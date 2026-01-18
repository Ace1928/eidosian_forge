from __future__ import annotations
import abc
import re
import os
import typing as T
from .base import DependencyException, DependencyMethods
from .configtool import ConfigToolDependency
from .detect import packages
from .framework import ExtraFrameworkDependency
from .pkgconfig import PkgConfigDependency
from .factory import DependencyFactory
from .. import mlog
from .. import mesonlib
def get_qt_winmain_base_name(self, is_debug: bool) -> str:
    return 'Qt6EntryPointd' if is_debug else 'Qt6EntryPoint'