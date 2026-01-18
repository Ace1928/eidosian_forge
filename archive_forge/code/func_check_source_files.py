from __future__ import annotations
import typing as T
from . import ExtensionModule, ModuleObject, MutableModuleObject, ModuleInfo
from .. import build
from .. import dependencies
from .. import mesonlib
from ..interpreterbase import (
from ..interpreterbase.decorators import ContainerTypeInfo, KwargInfo, typed_kwargs, typed_pos_args
from ..mesonlib import OrderedSet
def check_source_files(self, args: T.Sequence[T.Union[mesonlib.FileOrString, build.GeneratedTypes, dependencies.Dependency]]) -> T.Tuple[T.List[T.Union[mesonlib.FileOrString, build.GeneratedTypes]], T.List[dependencies.Dependency]]:
    sources: T.List[T.Union[mesonlib.FileOrString, build.GeneratedTypes]] = []
    deps: T.List[dependencies.Dependency] = []
    for x in args:
        if isinstance(x, dependencies.Dependency):
            deps.append(x)
        else:
            sources.append(x)
    to_check: T.List[str] = []
    for s in sources:
        if isinstance(s, str):
            to_check.append(s)
        elif isinstance(s, mesonlib.File):
            to_check.append(s.fname)
        else:
            to_check.extend(s.get_outputs())
    mesonlib.check_direntry_issues(to_check)
    return (sources, deps)