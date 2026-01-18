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
def kern(self) -> None:
    """
        Insert `Kern` nodes between `Char` nodes to set kerning.

        The `Char` nodes themselves determine the amount of kerning they need
        (in `~Char.get_kerning`), and this function just creates the correct
        linked list.
        """
    new_children = []
    num_children = len(self.children)
    if num_children:
        for i in range(num_children):
            elem = self.children[i]
            if i < num_children - 1:
                next = self.children[i + 1]
            else:
                next = None
            new_children.append(elem)
            kerning_distance = elem.get_kerning(next)
            if kerning_distance != 0.0:
                kern = Kern(kerning_distance)
                new_children.append(kern)
        self.children = new_children