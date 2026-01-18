from __future__ import annotations
import abc
import contextlib, os.path, re
import enum
import itertools
import typing as T
from functools import lru_cache
from .. import coredata
from .. import mlog
from .. import mesonlib
from ..mesonlib import (
from ..arglist import CompilerArgs
def get_base_compile_args(options: 'KeyedOptionDictType', compiler: 'Compiler') -> T.List[str]:
    args: T.List[str] = []
    try:
        if options[OptionKey('b_lto')].value:
            args.extend(compiler.get_lto_compile_args(threads=get_option_value(options, OptionKey('b_lto_threads'), 0), mode=get_option_value(options, OptionKey('b_lto_mode'), 'default')))
    except KeyError:
        pass
    try:
        args += compiler.get_colorout_args(options[OptionKey('b_colorout')].value)
    except KeyError:
        pass
    try:
        args += compiler.sanitizer_compile_args(options[OptionKey('b_sanitize')].value)
    except KeyError:
        pass
    try:
        pgo_val = options[OptionKey('b_pgo')].value
        if pgo_val == 'generate':
            args.extend(compiler.get_profile_generate_args())
        elif pgo_val == 'use':
            args.extend(compiler.get_profile_use_args())
    except KeyError:
        pass
    try:
        if options[OptionKey('b_coverage')].value:
            args += compiler.get_coverage_args()
    except KeyError:
        pass
    try:
        args += compiler.get_assert_args(are_asserts_disabled(options))
    except KeyError:
        pass
    if option_enabled(compiler.base_options, options, OptionKey('b_bitcode')):
        args.append('-fembed-bitcode')
    try:
        crt_val = options[OptionKey('b_vscrt')].value
        buildtype = options[OptionKey('buildtype')].value
        try:
            args += compiler.get_crt_compile_args(crt_val, buildtype)
        except AttributeError:
            pass
    except KeyError:
        pass
    return args