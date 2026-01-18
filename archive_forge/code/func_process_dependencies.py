from __future__ import annotations
import os, subprocess
import typing as T
from . import ExtensionModule, ModuleReturnValue, ModuleInfo
from .. import build, mesonlib, mlog
from ..build import CustomTarget, CustomTargetIndex
from ..dependencies import Dependency, InternalDependency
from ..interpreterbase import (
from ..interpreter.interpreterobjects import _CustomTargetHolder
from ..interpreter.type_checking import NoneType
from ..mesonlib import File, MesonException
from ..programs import ExternalProgram
def process_dependencies(self, deps: T.List[T.Union[Dependency, build.StaticLibrary, build.SharedLibrary, CustomTarget, CustomTargetIndex]]) -> T.List[str]:
    cflags = set()
    for dep in mesonlib.listify(ensure_list(deps)):
        if isinstance(dep, InternalDependency):
            inc_args = self.state.get_include_args(dep.include_directories)
            cflags.update([self.replace_dirs_in_string(x) for x in inc_args])
            cflags.update(self.process_dependencies(dep.libraries))
            cflags.update(self.process_dependencies(dep.sources))
            cflags.update(self.process_dependencies(dep.ext_deps))
        elif isinstance(dep, Dependency):
            cflags.update(dep.get_compile_args())
        elif isinstance(dep, (build.StaticLibrary, build.SharedLibrary)):
            self.extra_depends.append(dep)
            for incd in dep.get_include_dirs():
                cflags.update(incd.get_incdirs())
        elif isinstance(dep, HotdocTarget):
            self.process_dependencies(dep.get_target_dependencies())
            self._subprojects.extend(dep.subprojects)
            self.process_dependencies(dep.subprojects)
            self.include_paths.add(os.path.join(self.builddir, dep.hotdoc_conf.subdir))
            self.cmd += ['--extra-assets=' + p for p in dep.extra_assets]
            self.add_extension_paths(dep.extra_extension_paths)
        elif isinstance(dep, (CustomTarget, build.BuildTarget)):
            self.extra_depends.append(dep)
        elif isinstance(dep, CustomTargetIndex):
            self.extra_depends.append(dep.target)
    return [f.strip('-I') for f in cflags]