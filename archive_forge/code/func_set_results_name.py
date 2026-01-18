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
def set_results_name(self, name: str, list_all_matches: bool=False, *, listAllMatches: bool=False) -> 'ParserElement':
    """
        Define name for referencing matching tokens as a nested attribute
        of the returned parse results.

        Normally, results names are assigned as you would assign keys in a dict:
        any existing value is overwritten by later values. If it is necessary to
        keep all values captured for a particular results name, call ``set_results_name``
        with ``list_all_matches`` = True.

        NOTE: ``set_results_name`` returns a *copy* of the original :class:`ParserElement` object;
        this is so that the client can define a basic element, such as an
        integer, and reference it in multiple places with different names.

        You can also set results names using the abbreviated syntax,
        ``expr("name")`` in place of ``expr.set_results_name("name")``
        - see :class:`__call__`. If ``list_all_matches`` is required, use
        ``expr("name*")``.

        Example::

            date_str = (integer.set_results_name("year") + '/'
                        + integer.set_results_name("month") + '/'
                        + integer.set_results_name("day"))

            # equivalent form:
            date_str = integer("year") + '/' + integer("month") + '/' + integer("day")
        """
    listAllMatches = listAllMatches or list_all_matches
    return self._setResultsName(name, listAllMatches)