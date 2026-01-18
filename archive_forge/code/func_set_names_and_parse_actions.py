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
def set_names_and_parse_actions() -> None:
    for key, val in vars(p).items():
        if not key.startswith('_'):
            if key not in ('token', 'placeable', 'auto_delim'):
                val.setName(key)
            if hasattr(self, key):
                val.setParseAction(getattr(self, key))