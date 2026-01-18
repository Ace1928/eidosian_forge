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
@staticmethod
def get_pkgconfig_host_libexecs(core: PkgConfigDependency) -> str:
    return core.get_variable(pkgconfig='libexecdir')