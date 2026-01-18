from __future__ import annotations
import itertools
import os
import typing as T
from mesonbuild.interpreterbase.decorators import FeatureNew
from . import ExtensionModule, ModuleReturnValue, ModuleInfo
from .. import mesonlib, mlog
from ..build import (BothLibraries, BuildTarget, CustomTargetIndex, Executable, ExtractedObjects, GeneratedList,
from ..compilers.compilers import are_asserts_disabled, lang_suffixes
from ..interpreter.type_checking import (
from ..interpreterbase import ContainerTypeInfo, InterpreterException, KwargInfo, typed_kwargs, typed_pos_args, noPosargs, permittedKwargs
from ..mesonlib import File
from ..programs import ExternalProgram
@FeatureNew('rust.proc_macro', '1.3.0')
@permittedKwargs({'rust_args', 'rust_dependency_map', 'sources', 'dependencies', 'extra_files', 'link_args', 'link_depends', 'link_with', 'override_options'})
@typed_pos_args('rust.proc_macro', str, varargs=SOURCES_VARARGS)
@typed_kwargs('rust.proc_macro', *SHARED_LIB_KWS, allow_unknown=True)
def proc_macro(self, state: ModuleState, args: T.Tuple[str, SourcesVarargsType], kwargs: _kwargs.SharedLibrary) -> SharedLibrary:
    kwargs['native'] = True
    kwargs['rust_crate_type'] = 'proc-macro'
    kwargs['rust_args'] = kwargs['rust_args'] + ['--extern', 'proc_macro']
    target = state._interpreter.build_target(state.current_node, args, kwargs, SharedLibrary)
    return target