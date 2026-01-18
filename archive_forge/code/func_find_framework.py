from __future__ import annotations
import collections
import functools
import glob
import itertools
import os
import re
import subprocess
import copy
import typing as T
from pathlib import Path
from ... import arglist
from ... import mesonlib
from ... import mlog
from ...linkers.linkers import GnuLikeDynamicLinkerMixin, SolarisDynamicLinker, CompCertDynamicLinker
from ...mesonlib import LibType, OptionKey
from .. import compilers
from ..compilers import CompileCheckMode
from .visualstudio import VisualStudioLikeCompiler
def find_framework(self, name: str, env: 'Environment', extra_dirs: T.List[str], allow_system: bool=True) -> T.Optional[T.List[str]]:
    """
        Finds the framework with the specified name, and returns link args for
        the same or returns None when the framework is not found.
        """
    return self._find_framework_impl(name, env, extra_dirs, allow_system)