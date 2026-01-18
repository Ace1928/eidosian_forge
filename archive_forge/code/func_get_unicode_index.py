from __future__ import annotations
import abc
import copy
import enum
import functools
import logging
import os
import re
import types
import unicodedata
import string
import typing as T
from typing import NamedTuple
import numpy as np
from pyparsing import (
import matplotlib as mpl
from . import cbook
from ._mathtext_data import (
from .font_manager import FontProperties, findfont, get_font
from .ft2font import FT2Font, FT2Image, KERNING_DEFAULT
from packaging.version import parse as parse_version
from pyparsing import __version__ as pyparsing_version
def get_unicode_index(symbol: str) -> int:
    """
    Return the integer index (from the Unicode table) of *symbol*.

    Parameters
    ----------
    symbol : str
        A single (Unicode) character, a TeX command (e.g. r'\\pi') or a Type1
        symbol name (e.g. 'phi').
    """
    try:
        return ord(symbol)
    except TypeError:
        pass
    try:
        return tex2uni[symbol.strip('\\')]
    except KeyError as err:
        raise ValueError(f'{symbol!r} is not a valid Unicode character or TeX/Type1 symbol') from err