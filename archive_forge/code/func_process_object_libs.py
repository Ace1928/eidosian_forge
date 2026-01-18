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
def process_object_libs(self, obj_target_list: T.List['ConverterTarget'], linker_workaround: bool) -> None:
    temp = [x for x in self.generated if any((x.name.endswith('.' + y) for y in obj_suffixes))]
    stem = [x.stem for x in temp]
    exts = self._all_source_suffixes()
    for i in obj_target_list:
        source_files = [x.name for x in i.sources + i.generated]
        for j in stem:
            candidates = [j]
            if not any((j.endswith('.' + x) for x in exts)):
                mlog.warning('Object files do not contain source file extensions, thus falling back to guessing them.', once=True)
                candidates += [f'{j}.{x}' for x in exts]
            if any((x in source_files for x in candidates)):
                if linker_workaround:
                    self._append_objlib_sources(i)
                else:
                    self.includes += i.includes
                    self.includes = list(OrderedSet(self.includes))
                    self.object_libs += [i]
                break
    self.generated = [x for x in self.generated if not any((x.name.endswith('.' + y) for y in obj_suffixes))]