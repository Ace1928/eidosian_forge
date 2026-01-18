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
def set_debug_actions(self, start_action: DebugStartAction, success_action: DebugSuccessAction, exception_action: DebugExceptionAction) -> 'ParserElement':
    """
        Customize display of debugging messages while doing pattern matching:

        - ``start_action`` - method to be called when an expression is about to be parsed;
          should have the signature ``fn(input_string: str, location: int, expression: ParserElement, cache_hit: bool)``

        - ``success_action`` - method to be called when an expression has successfully parsed;
          should have the signature ``fn(input_string: str, start_location: int, end_location: int, expression: ParserELement, parsed_tokens: ParseResults, cache_hit: bool)``

        - ``exception_action`` - method to be called when expression fails to parse;
          should have the signature ``fn(input_string: str, location: int, expression: ParserElement, exception: Exception, cache_hit: bool)``
        """
    self.debugActions = self.DebugActions(start_action or _default_start_debug_action, success_action or _default_success_debug_action, exception_action or _default_exception_debug_action)
    self.debug = True
    return self