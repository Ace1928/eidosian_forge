from __future__ import annotations
import abc
import os
import typing as T
import re
from .base import ArLikeLinker, RSPFileSyntax
from .. import mesonlib
from ..mesonlib import EnvironmentException, MesonException
from ..arglist import CompilerArgs
def headerpad_args(self) -> T.List[str]:
    return self._apply_prefix('-headerpad_max_install_names')