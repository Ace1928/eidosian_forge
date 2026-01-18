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
def operatorname(self, s: str, loc: int, toks: ParseResults) -> T.Any:
    self.push_state()
    state = self.get_state()
    state.font = 'rm'
    hlist_list: list[Node] = []
    name = toks['name']
    for c in name:
        if isinstance(c, Char):
            c.font = 'rm'
            c._update_metrics()
            hlist_list.append(c)
        elif isinstance(c, str):
            hlist_list.append(Char(c, state))
        else:
            hlist_list.append(c)
    next_char_loc = loc + len(name) + 1
    if isinstance(name, ParseResults):
        next_char_loc += len('operatorname{}')
    next_char = next((c for c in s[next_char_loc:] if c != ' '), '')
    delimiters = self._delims | {'^', '_'}
    if next_char not in delimiters and name not in self._overunder_functions:
        hlist_list += [self._make_space(self._space_widths['\\,'])]
    self.pop_state()
    if next_char in {'^', '_'}:
        self._in_subscript_or_superscript = True
    else:
        self._in_subscript_or_superscript = False
    return Hlist(hlist_list)