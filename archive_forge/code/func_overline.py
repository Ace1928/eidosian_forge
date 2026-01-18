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
def overline(self, toks: ParseResults) -> T.Any:
    body = toks['body']
    state = self.get_state()
    thickness = state.get_current_underline_thickness()
    height = body.height - body.shift_amount + thickness * 3.0
    depth = body.depth + body.shift_amount
    rightside = Vlist([Hrule(state), Glue('fill'), Hlist([body])])
    rightside.vpack(height + state.fontsize * state.dpi / (100.0 * 12.0), 'exactly', depth)
    hlist = Hlist([rightside])
    return [hlist]