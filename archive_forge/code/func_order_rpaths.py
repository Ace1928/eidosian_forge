from __future__ import annotations
import abc
import os
import typing as T
import re
from .base import ArLikeLinker, RSPFileSyntax
from .. import mesonlib
from ..mesonlib import EnvironmentException, MesonException
from ..arglist import CompilerArgs
def order_rpaths(rpath_list: T.List[str]) -> T.List[str]:
    return sorted(rpath_list, key=os.path.isabs)