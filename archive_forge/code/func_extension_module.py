from __future__ import annotations
import sysconfig
import typing as T
from .. import mesonlib
from . import ExtensionModule, ModuleInfo, ModuleState
from ..build import (
from ..interpreter.type_checking import SHARED_MOD_KWS
from ..interpreterbase import typed_kwargs, typed_pos_args, noPosargs, noKwargs, permittedKwargs
from ..programs import ExternalProgram
@permittedKwargs(known_shmod_kwargs - {'name_prefix', 'name_suffix'})
@typed_pos_args('python3.extension_module', str, varargs=(str, mesonlib.File, CustomTarget, CustomTargetIndex, GeneratedList, StructuredSources, ExtractedObjects, BuildTarget))
@typed_kwargs('python3.extension_module', *_MOD_KWARGS, allow_unknown=True)
def extension_module(self, state: ModuleState, args: T.Tuple[str, T.List[BuildTargetSource]], kwargs: SharedModuleKW):
    host_system = state.environment.machines.host.system
    if host_system == 'darwin':
        suffix = 'so'
    elif host_system == 'windows':
        suffix = 'pyd'
    else:
        suffix = []
    kwargs['name_prefix'] = ''
    kwargs['name_suffix'] = suffix
    return self.interpreter.build_target(state.current_node, args, kwargs, SharedModule)