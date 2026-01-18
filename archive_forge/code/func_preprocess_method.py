from __future__ import annotations
import collections
import enum
import functools
import os
import itertools
import typing as T
from .. import build
from .. import coredata
from .. import dependencies
from .. import mesonlib
from .. import mlog
from ..compilers import SUFFIX_TO_LANG
from ..compilers.compilers import CompileCheckMode
from ..interpreterbase import (ObjectHolder, noPosargs, noKwargs,
from ..interpreterbase.decorators import ContainerTypeInfo, typed_kwargs, KwargInfo, typed_pos_args
from ..mesonlib import OptionKey
from .interpreterobjects import (extract_required_kwarg, extract_search_dirs)
from .type_checking import REQUIRED_KW, in_set_validator, NoneType
@FeatureNew('compiler.preprocess', '0.64.0')
@FeatureNewKwargs('compiler.preprocess', '1.3.2', ['compile_args'], extra_message='compile_args were ignored before this version')
@typed_pos_args('compiler.preprocess', varargs=(str, mesonlib.File, build.CustomTarget, build.CustomTargetIndex, build.GeneratedList), min_varargs=1)
@typed_kwargs('compiler.preprocess', KwargInfo('output', str, default='@PLAINNAME@.i'), KwargInfo('compile_args', ContainerTypeInfo(list, str), listify=True, default=[]), _INCLUDE_DIRS_KW, _DEPENDENCIES_KW.evolve(since='1.1.0'), _DEPENDS_KW.evolve(since='1.4.0'))
def preprocess_method(self, args: T.Tuple[T.List['mesonlib.FileOrString']], kwargs: 'PreprocessKW') -> T.List[build.CustomTargetIndex]:
    compiler = self.compiler.get_preprocessor()
    _sources: T.List[mesonlib.File] = self.interpreter.source_strings_to_files(args[0])
    sources = T.cast('T.List[SourceOutputs]', _sources)
    if any((isinstance(s, (build.CustomTarget, build.CustomTargetIndex, build.GeneratedList)) for s in sources)):
        FeatureNew.single_use('compiler.preprocess with generated sources', '1.1.0', self.subproject, location=self.current_node)
    tg_counter = next(self.preprocess_uid[self.interpreter.subdir])
    if tg_counter > 0:
        FeatureNew.single_use('compiler.preprocess used multiple times', '1.1.0', self.subproject, location=self.current_node)
    tg_name = f'preprocessor_{tg_counter}'
    tg = build.CompileTarget(tg_name, self.interpreter.subdir, self.subproject, self.environment, sources, kwargs['output'], compiler, self.interpreter.backend, kwargs['compile_args'], kwargs['include_directories'], kwargs['dependencies'], kwargs['depends'])
    self.interpreter.add_target(tg.name, tg)
    private_dir = os.path.relpath(self.interpreter.backend.get_target_private_dir(tg), self.interpreter.subdir)
    return [build.CustomTargetIndex(tg, os.path.join(private_dir, o)) for o in tg.outputs]