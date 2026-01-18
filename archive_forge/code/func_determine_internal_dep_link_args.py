from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def determine_internal_dep_link_args(self, target, buildtype):
    links_dylib = False
    dep_libs = []
    for l in target.link_targets:
        if isinstance(target, build.SharedModule) and isinstance(l, build.Executable):
            continue
        if isinstance(l, build.CustomTargetIndex):
            rel_dir = self.get_custom_target_output_dir(l.target)
            libname = l.get_filename()
        elif isinstance(l, build.CustomTarget):
            rel_dir = self.get_custom_target_output_dir(l)
            libname = l.get_filename()
        else:
            rel_dir = self.get_target_dir(l)
            libname = l.get_filename()
        abs_path = os.path.join(self.environment.get_build_dir(), rel_dir, libname)
        dep_libs.append("'%s'" % abs_path)
        if isinstance(l, build.SharedLibrary):
            links_dylib = True
        if isinstance(l, build.StaticLibrary):
            sub_libs, sub_links_dylib = self.determine_internal_dep_link_args(l, buildtype)
            dep_libs += sub_libs
            links_dylib = links_dylib or sub_links_dylib
    return (dep_libs, links_dylib)