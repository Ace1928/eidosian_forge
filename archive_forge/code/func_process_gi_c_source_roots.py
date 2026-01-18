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
def process_gi_c_source_roots(self) -> None:
    if self.hotdoc.run_hotdoc(['--has-extension=gi-extension']) != 0:
        return
    value = self.kwargs.pop('gi_c_source_roots')
    value.extend([os.path.join(self.sourcedir, self.state.root_subdir), os.path.join(self.builddir, self.state.root_subdir)])
    self.cmd += ['--gi-c-source-roots'] + value