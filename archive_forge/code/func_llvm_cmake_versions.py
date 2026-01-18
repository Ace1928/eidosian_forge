from __future__ import annotations
import glob
import os
import re
import pathlib
import shutil
import subprocess
import typing as T
import functools
from mesonbuild.interpreterbase.decorators import FeatureDeprecated
from .. import mesonlib, mlog
from ..environment import get_llvm_tool_names
from ..mesonlib import version_compare, version_compare_many, search_version, stringlistify, extract_as_list
from .base import DependencyException, DependencyMethods, detect_compiler, strip_system_includedirs, strip_system_libdirs, SystemDependency, ExternalDependency, DependencyTypeName
from .cmake import CMakeDependency
from .configtool import ConfigToolDependency
from .detect import packages
from .factory import DependencyFactory
from .misc import threads_factory
from .pkgconfig import PkgConfigDependency
def llvm_cmake_versions(self) -> T.List[str]:

    def ver_from_suf(req: str) -> str:
        return search_version(req.strip('-') + '.0')

    def version_sorter(a: str, b: str) -> int:
        if version_compare(a, '=' + b):
            return 0
        if version_compare(a, '<' + b):
            return 1
        return -1
    llvm_requested_versions = [ver_from_suf(x) for x in get_llvm_tool_names('') if version_compare(ver_from_suf(x), '>=0')]
    if self.version_reqs:
        llvm_requested_versions = [ver_from_suf(x) for x in get_llvm_tool_names('') if version_compare_many(ver_from_suf(x), self.version_reqs)]
    return sorted(llvm_requested_versions, key=functools.cmp_to_key(version_sorter))