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
def search_string(self, instring: str, max_matches: int=_MAX_INT, *, debug: bool=False, maxMatches: int=_MAX_INT) -> ParseResults:
    """
        Another extension to :class:`scan_string`, simplifying the access to the tokens found
        to match the given parse expression.  May be called with optional
        ``max_matches`` argument, to clip searching after 'n' matches are found.

        Example::

            # a capitalized word starts with an uppercase letter, followed by zero or more lowercase letters
            cap_word = Word(alphas.upper(), alphas.lower())

            print(cap_word.search_string("More than Iron, more than Lead, more than Gold I need Electricity"))

            # the sum() builtin can be used to merge results into a single ParseResults object
            print(sum(cap_word.search_string("More than Iron, more than Lead, more than Gold I need Electricity")))

        prints::

            [['More'], ['Iron'], ['Lead'], ['Gold'], ['I'], ['Electricity']]
            ['More', 'Iron', 'Lead', 'Gold', 'I', 'Electricity']
        """
    maxMatches = min(maxMatches, max_matches)
    try:
        return ParseResults([t for t, s, e in self.scan_string(instring, maxMatches, debug=debug)])
    except ParseBaseException as exc:
        if ParserElement.verbose_stacktrace:
            raise
        else:
            raise exc.with_traceback(None)