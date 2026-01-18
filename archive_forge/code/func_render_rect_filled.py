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
def render_rect_filled(self, output: Output, x1: float, y1: float, x2: float, y2: float) -> None:
    """
        Draw a filled rectangle from (*x1*, *y1*) to (*x2*, *y2*).
        """
    output.rects.append((x1, y1, x2, y2))