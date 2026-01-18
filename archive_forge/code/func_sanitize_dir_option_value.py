from __future__ import annotations
import copy
from . import mlog, mparser
import pickle, os, uuid
import sys
from itertools import chain
from pathlib import PurePath
from collections import OrderedDict, abc
from dataclasses import dataclass
from .mesonlib import (
from .wrap import WrapMode
import ast
import argparse
import configparser
import enum
import shlex
import typing as T
def sanitize_dir_option_value(self, prefix: str, option: OptionKey, value: T.Any) -> T.Any:
    """
        If the option is an installation directory option, the value is an
        absolute path and resides within prefix, return the value
        as a path relative to the prefix. Otherwise, return it as is.

        This way everyone can do f.ex, get_option('libdir') and usually get
        the library directory relative to prefix, even though it really
        should not be relied upon.
        """
    try:
        value = PurePath(value)
    except TypeError:
        return value
    if option.name.endswith('dir') and value.is_absolute() and (option not in BUILTIN_DIR_NOPREFIX_OPTIONS):
        try:
            value = value.relative_to(prefix)
        except ValueError:
            pass
        if '..' in value.parts:
            raise MesonException(f"The value of the '{option}' option is '{value}' but directory options are not allowed to contain '..'.\nIf you need a path outside of the {prefix!r} prefix, please use an absolute path.")
    return value.as_posix()