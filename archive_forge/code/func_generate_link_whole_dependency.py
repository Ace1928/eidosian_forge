from __future__ import annotations
import copy
import os
import collections
import itertools
import typing as T
from enum import Enum
from .. import mlog, mesonlib
from ..compilers import clib_langs
from ..mesonlib import LibType, MachineChoice, MesonException, HoldableObject, OptionKey
from ..mesonlib import version_compare_many
def generate_link_whole_dependency(self) -> Dependency:
    from ..build import SharedLibrary, CustomTarget, CustomTargetIndex
    new_dep = copy.deepcopy(self)
    for x in new_dep.libraries:
        if isinstance(x, SharedLibrary):
            raise MesonException('Cannot convert a dependency to link_whole when it contains a SharedLibrary')
        elif isinstance(x, (CustomTarget, CustomTargetIndex)) and x.links_dynamically():
            raise MesonException('Cannot convert a dependency to link_whole when it contains a CustomTarget or CustomTargetIndex which is a shared library')
    new_dep.whole_libraries += T.cast('T.List[T.Union[StaticLibrary, CustomTarget, CustomTargetIndex]]', new_dep.libraries)
    new_dep.libraries = []
    return new_dep