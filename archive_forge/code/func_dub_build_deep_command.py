from __future__ import annotations
from .base import ExternalDependency, DependencyException, DependencyTypeName
from .pkgconfig import PkgConfigDependency
from ..mesonlib import (Popen_safe, OptionKey, join_args, version_compare)
from ..programs import ExternalProgram
from .. import mlog
import re
import os
import json
import typing as T
def dub_build_deep_command() -> str:
    cmd = ['dub', 'run', 'dub-build-deep', '--yes', '--', main_pack_spec, '--arch=' + dub_arch, '--compiler=' + self.compiler.get_exelist()[-1], '--build=' + dub_buildtype]
    return join_args(cmd)