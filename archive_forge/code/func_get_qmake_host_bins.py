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
def get_qmake_host_bins(qvars: T.Dict[str, str]) -> str:
    if 'QT_HOST_BINS' in qvars:
        return qvars['QT_HOST_BINS']
    return qvars['QT_INSTALL_BINS']