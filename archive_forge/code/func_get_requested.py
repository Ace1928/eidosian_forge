from __future__ import annotations
import glob
import re
import os
import typing as T
from pathlib import Path
from .. import mesonlib
from .. import mlog
from ..environment import detect_cpu_family
from .base import DependencyException, SystemDependency
from .detect import packages
def get_requested(self, kwargs: T.Dict[str, T.Any]) -> T.List[str]:
    candidates = mesonlib.extract_as_list(kwargs, 'modules')
    for c in candidates:
        if not isinstance(c, str):
            raise DependencyException('CUDA module argument is not a string.')
    return candidates