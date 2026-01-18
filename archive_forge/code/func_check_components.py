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
def check_components(self, modules: T.List[str], required: bool=True) -> None:
    """Check for llvm components (modules in meson terms).

        The required option is whether the module is required, not whether LLVM
        is required.
        """
    for mod in sorted(set(modules)):
        status = ''
        if mod not in self.provided_modules:
            if required:
                self.is_found = False
                if self.required:
                    raise DependencyException(f'Could not find required LLVM Component: {mod}')
                status = '(missing)'
            else:
                status = '(missing but optional)'
        else:
            self.required_modules.add(mod)
        self.module_details.append(mod + status)