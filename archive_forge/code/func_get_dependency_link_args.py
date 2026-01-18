from __future__ import annotations
import os.path
import typing as T
from ... import coredata
from ... import mesonlib
from ...mesonlib import OptionKey
from ...mesonlib import LibType
from mesonbuild.compilers.compilers import CompileCheckMode
def get_dependency_link_args(self, dep: 'Dependency') -> T.List[str]:
    return wrap_js_includes(super().get_dependency_link_args(dep))