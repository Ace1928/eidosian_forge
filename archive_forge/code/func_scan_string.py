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
def scan_string(self, instring: str, max_matches: int=_MAX_INT, overlap: bool=False, *, debug: bool=False, maxMatches: int=_MAX_INT) -> Generator[Tuple[ParseResults, int, int], None, None]:
    """
        Scan the input string for expression matches.  Each match will return the
        matching tokens, start location, and end location.  May be called with optional
        ``max_matches`` argument, to clip scanning after 'n' matches are found.  If
        ``overlap`` is specified, then overlapping matches will be reported.

        Note that the start and end locations are reported relative to the string
        being parsed.  See :class:`parse_string` for more information on parsing
        strings with embedded tabs.

        Example::

            source = "sldjf123lsdjjkf345sldkjf879lkjsfd987"
            print(source)
            for tokens, start, end in Word(alphas).scan_string(source):
                print(' '*start + '^'*(end-start))
                print(' '*start + tokens[0])

        prints::

            sldjf123lsdjjkf345sldkjf879lkjsfd987
            ^^^^^
            sldjf
                    ^^^^^^^
                    lsdjjkf
                              ^^^^^^
                              sldkjf
                                       ^^^^^^
                                       lkjsfd
        """
    maxMatches = min(maxMatches, max_matches)
    if not self.streamlined:
        self.streamline()
    for e in self.ignoreExprs:
        e.streamline()
    if not self.keepTabs:
        instring = str(instring).expandtabs()
    instrlen = len(instring)
    loc = 0
    preparseFn = self.preParse
    parseFn = self._parse
    ParserElement.resetCache()
    matches = 0
    try:
        while loc <= instrlen and matches < maxMatches:
            try:
                preloc: int = preparseFn(instring, loc)
                nextLoc: int
                tokens: ParseResults
                nextLoc, tokens = parseFn(instring, preloc, callPreParse=False)
            except ParseException:
                loc = preloc + 1
            else:
                if nextLoc > loc:
                    matches += 1
                    if debug:
                        print({'tokens': tokens.asList(), 'start': preloc, 'end': nextLoc})
                    yield (tokens, preloc, nextLoc)
                    if overlap:
                        nextloc = preparseFn(instring, loc)
                        if nextloc > loc:
                            loc = nextLoc
                        else:
                            loc += 1
                    else:
                        loc = nextLoc
                else:
                    loc = preloc + 1
    except ParseBaseException as exc:
        if ParserElement.verbose_stacktrace:
            raise
        else:
            raise exc.with_traceback(None)