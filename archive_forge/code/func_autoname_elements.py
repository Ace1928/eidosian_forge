from collections import deque
import os
import typing
from typing import (
from abc import ABC, abstractmethod
from enum import Enum
import string
import copy
import warnings
import re
import sys
from collections.abc import Iterable
import traceback
import types
from operator import itemgetter
from functools import wraps
from threading import RLock
from pathlib import Path
from .util import (
from .exceptions import *
from .actions import *
from .results import ParseResults, _ParseResultsWithOffset
from .unicode import pyparsing_unicode
def autoname_elements() -> None:
    """
    Utility to simplify mass-naming of parser elements, for
    generating railroad diagram with named subdiagrams.
    """
    calling_frame = sys._getframe().f_back
    if calling_frame is None:
        return
    calling_frame = typing.cast(types.FrameType, calling_frame)
    for name, var in calling_frame.f_locals.items():
        if isinstance(var, ParserElement) and (not var.customName):
            var.set_name(name)