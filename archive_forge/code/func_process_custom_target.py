from __future__ import annotations
from functools import lru_cache
from os import environ
from pathlib import Path
import re
import typing as T
from .common import CMakeException, CMakeTarget, language_map, cmake_get_generator_args, check_cmake_args
from .fileapi import CMakeFileAPI
from .executor import CMakeExecutor
from .toolchain import CMakeToolchain, CMakeExecScope
from .traceparser import CMakeTraceParser
from .tracetargets import resolve_cmake_trace_targets
from .. import mlog, mesonlib
from ..mesonlib import MachineChoice, OrderedSet, path_is_in_root, relative_to_if_possible, OptionKey
from ..mesondata import DataFile
from ..compilers.compilers import assembler_suffixes, lang_suffixes, header_suffixes, obj_suffixes, lib_suffixes, is_header
from ..programs import ExternalProgram
from ..coredata import FORBIDDEN_TARGET_NAMES
from ..mparser import (
def process_custom_target(tgt: ConverterCustomTarget) -> None:
    detect_cycle(tgt)
    tgt_var = tgt.name

    def resolve_source(x: T.Union[str, ConverterTarget, ConverterCustomTarget, CustomTargetReference]) -> T.Union[str, IdNode, IndexNode]:
        if isinstance(x, ConverterTarget):
            if x.name not in processed:
                process_target(x)
            return extract_tgt(x)
        if isinstance(x, ConverterCustomTarget):
            if x.name not in processed:
                process_custom_target(x)
            return extract_tgt(x)
        elif isinstance(x, CustomTargetReference):
            if x.ctgt.name not in processed:
                process_custom_target(x.ctgt)
            return resolve_ctgt_ref(x)
        else:
            return x
    command: T.List[T.Union[str, IdNode, IndexNode]] = []
    command += mesonlib.get_meson_command()
    command += ['--internal', 'cmake_run_ctgt']
    command += ['-o', '@OUTPUT@']
    if tgt.original_outputs:
        command += ['-O'] + [x.as_posix() for x in tgt.original_outputs]
    command += ['-d', tgt.working_dir.as_posix()]
    for cmd in tgt.command:
        command += [resolve_source(x) for x in cmd] + [';;;']
    tgt_kwargs: TYPE_mixed_kwargs = {'input': [resolve_source(x) for x in tgt.inputs], 'output': tgt.outputs, 'command': command, 'depends': [resolve_source(x) for x in tgt.depends]}
    root_cb.lines += [assign(tgt_var, function('custom_target', [tgt.name], tgt_kwargs))]
    processed[tgt.name] = {'inc': None, 'src': None, 'dep': None, 'tgt': tgt_var, 'func': 'custom_target'}
    name_map[tgt.cmake_name] = tgt.name