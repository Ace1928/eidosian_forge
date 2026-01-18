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
def add_parse_action(self, *fns: ParseAction, **kwargs) -> 'ParserElement':
    """
        Add one or more parse actions to expression's list of parse actions. See :class:`set_parse_action`.

        See examples in :class:`copy`.
        """
    self.parseAction += [_trim_arity(fn) for fn in fns]
    self.callDuringTry = self.callDuringTry or kwargs.get('call_during_try', kwargs.get('callDuringTry', False))
    return self