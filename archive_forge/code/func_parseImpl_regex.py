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
def parseImpl_regex(self, instring, loc, doActions=True):
    result = self.re_match(instring, loc)
    if not result:
        raise ParseException(instring, loc, self.errmsg, self)
    loc = result.end()
    return (loc, result.group())