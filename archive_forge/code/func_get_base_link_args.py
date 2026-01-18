from __future__ import annotations
import abc
import os
import typing as T
import re
from .base import ArLikeLinker, RSPFileSyntax
from .. import mesonlib
from ..mesonlib import EnvironmentException, MesonException
from ..arglist import CompilerArgs
def get_base_link_args(self, options: 'KeyedOptionDictType') -> T.List[str]:
    """Like compilers.get_base_link_args, but for the static linker."""
    return []