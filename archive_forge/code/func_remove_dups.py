from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePath
import os
import typing as T
from . import NewExtensionModule, ModuleInfo
from . import ModuleReturnValue
from .. import build
from .. import dependencies
from .. import mesonlib
from .. import mlog
from ..coredata import BUILTIN_DIR_OPTIONS
from ..dependencies.pkgconfig import PkgConfigDependency, PkgConfigInterface
from ..interpreter.type_checking import D_MODULE_VERSIONS_KW, INSTALL_DIR_KW, VARIABLES_KW, NoneType
from ..interpreterbase import FeatureNew, FeatureDeprecated
from ..interpreterbase.decorators import ContainerTypeInfo, KwargInfo, typed_kwargs, typed_pos_args
def remove_dups(self) -> None:
    exclude: T.Set[str] = set()

    def _ids(x: T.Union[str, build.CustomTarget, build.CustomTargetIndex, build.StaticLibrary, build.SharedLibrary]) -> T.Iterable[str]:
        if isinstance(x, str):
            yield x
        else:
            if x.get_id() in self.metadata:
                yield self.metadata[x.get_id()].display_name
            yield x.get_id()

    def _add_exclude(x: T.Union[str, build.CustomTarget, build.CustomTargetIndex, build.StaticLibrary, build.SharedLibrary]) -> bool:
        was_excluded = False
        for i in _ids(x):
            if i in exclude:
                was_excluded = True
            else:
                exclude.add(i)
        return was_excluded
    for t in self.link_whole_targets:
        _add_exclude(t)

    @T.overload
    def _fn(xs: T.List[str], libs: bool=False) -> T.List[str]:
        ...

    @T.overload
    def _fn(xs: T.List[LIBS], libs: bool=False) -> T.List[LIBS]:
        ...

    def _fn(xs: T.Union[T.List[str], T.List[LIBS]], libs: bool=False) -> T.Union[T.List[str], T.List[LIBS]]:
        result = []
        for x in xs:
            known_flags = ['-pthread']
            cannot_dedup = libs and isinstance(x, str) and (not x.startswith(('-l', '-L'))) and (x not in known_flags)
            if not cannot_dedup and _add_exclude(x):
                continue
            result.append(x)
        return result
    self.pub_reqs = _fn(self.pub_reqs)
    self.pub_libs = _fn(self.pub_libs, True)
    self.priv_reqs = _fn(self.priv_reqs)
    self.priv_libs = _fn(self.priv_libs, True)
    exclude = set()
    self.cflags = _fn(self.cflags)