from __future__ import annotations
import abc
import os
import typing as T
from ... import arglist
from ... import mesonlib
from ... import mlog
from mesonbuild.compilers.compilers import CompileCheckMode
def get_preprocess_only_args(self) -> T.List[str]:
    return ['/EP']