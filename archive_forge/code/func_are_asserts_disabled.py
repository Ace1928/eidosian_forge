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
def are_asserts_disabled(options: KeyedOptionDictType) -> bool:
    """Should debug assertions be disabled

    :param options: OptionDictionary
    :return: whether to disable assertions or not
    """
    return options[OptionKey('b_ndebug')].value == 'true' or (options[OptionKey('b_ndebug')].value == 'if-release' and options[OptionKey('buildtype')].value in {'release', 'plain'})