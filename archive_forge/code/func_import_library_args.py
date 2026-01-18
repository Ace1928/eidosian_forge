from __future__ import annotations
import abc
import os
import typing as T
import re
from .base import ArLikeLinker, RSPFileSyntax
from .. import mesonlib
from ..mesonlib import EnvironmentException, MesonException
from ..arglist import CompilerArgs
def import_library_args(self, implibname: str) -> T.List[str]:
    """The command to generate the import library."""
    return self._apply_prefix(['/IMPLIB:' + implibname])