from __future__ import annotations
import os
import typing as T
from .. import mesonlib
from .. import dependencies
from .. import build
from .. import mlog, coredata
from ..mesonlib import MachineChoice, OptionKey
from ..programs import OverrideProgram, ExternalProgram
from ..interpreter.type_checking import ENV_KW, ENV_METHOD_KW, ENV_SEPARATOR_KW, env_convertor_with_method
from ..interpreterbase import (MesonInterpreterObject, FeatureNew, FeatureDeprecated,
from .primitives import MesonVersionString
from .type_checking import NATIVE_KW, NoneType
@FeatureNew('meson.override_find_program', '0.46.0')
@typed_pos_args('meson.override_find_program', str, (mesonlib.File, ExternalProgram, build.Executable))
@noKwargs
def override_find_program_method(self, args: T.Tuple[str, T.Union[mesonlib.File, ExternalProgram, build.Executable]], kwargs: 'TYPE_kwargs') -> None:
    name, exe = args
    if isinstance(exe, mesonlib.File):
        abspath = exe.absolute_path(self.interpreter.environment.source_dir, self.interpreter.environment.build_dir)
        if not os.path.exists(abspath):
            raise InterpreterException(f'Tried to override {name} with a file that does not exist.')
        exe = OverrideProgram(name, [abspath])
    self.interpreter.add_find_program_override(name, exe)