import os
import re
import sys
import traceback
import ast
import importlib
from re import sub, findall
from types import CodeType
from functools import partial
from collections import OrderedDict, defaultdict
import kivy.lang.builder  # imported as absolute to avoid circular import
from kivy.logger import Logger
from kivy.cache import Cache
from kivy import require
from kivy.resources import resource_find
from kivy.utils import rgba
import kivy.metrics as Metrics
def strip_comments(self, lines):
    """Remove all comments from all lines in-place.
           Comments need to be on a single line and not at the end of a line.
           i.e. a comment line's first non-whitespace character must be a #.
        """
    for ln, line in lines[:]:
        stripped = line.strip()
        if stripped[:2] == '#:':
            self.directives.append((ln, stripped[2:]))
        if stripped[:1] == '#':
            lines.remove((ln, line))
        if not stripped:
            lines.remove((ln, line))