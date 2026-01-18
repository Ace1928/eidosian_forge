from __future__ import annotations
from .base import ExternalDependency, DependencyException, DependencyTypeName
from ..mesonlib import listify, Popen_safe, Popen_safe_logged, split_args, version_compare, version_compare_many
from ..programs import find_external_program
from .. import mlog
import re
import typing as T
from mesonbuild import mesonlib
def get_variable_args(self, variable_name: str) -> T.List[str]:
    return [f'--{variable_name}']