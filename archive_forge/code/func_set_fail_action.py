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
def set_fail_action(self, fn: ParseFailAction) -> 'ParserElement':
    """
        Define action to perform if parsing fails at this expression.
        Fail acton fn is a callable function that takes the arguments
        ``fn(s, loc, expr, err)`` where:

        - ``s`` = string being parsed
        - ``loc`` = location where expression match was attempted and failed
        - ``expr`` = the parse expression that failed
        - ``err`` = the exception thrown

        The function returns no value.  It may throw :class:`ParseFatalException`
        if it is desired to stop parsing immediately."""
    self.failAction = fn
    return self