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
def get_xheight(self, fontname: str, fontsize: float, dpi: float) -> float:
    font = self._get_font(fontname)
    font.set_size(fontsize, dpi)
    pclt = font.get_sfnt_table('pclt')
    if pclt is None:
        metrics = self.get_metrics(fontname, mpl.rcParams['mathtext.default'], 'x', fontsize, dpi)
        return metrics.iceberg
    xHeight = pclt['xHeight'] / 64.0 * (fontsize / 12.0) * (dpi / 100.0)
    return xHeight