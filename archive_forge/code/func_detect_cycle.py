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
def detect_cycle(tgt: T.Union[ConverterTarget, ConverterCustomTarget]) -> None:
    if tgt.name in processing:
        raise CMakeException('Cycle in CMake inputs/dependencies detected')
    processing.append(tgt.name)