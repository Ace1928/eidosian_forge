import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def scanString(self, instring, maxMatches=_MAX_INT, overlap=False):
    """Scan the input string for expression matches.  Each match will return the
           matching tokens, start location, and end location.  May be called with optional
           C{maxMatches} argument, to clip scanning after 'n' matches are found.  If
           C{overlap} is specified, then overlapping matches will be reported.

           Note that the start and end locations are reported relative to the string
           being parsed.  See L{I{parseString}<parseString>} for more information on parsing
           strings with embedded tabs."""
    if not self.streamlined:
        self.streamline()
    for e in self.ignoreExprs:
        e.streamline()
    if not self.keepTabs:
        instring = _ustr(instring).expandtabs()
    instrlen = len(instring)
    loc = 0
    preparseFn = self.preParse
    parseFn = self._parse
    ParserElement.resetCache()
    matches = 0
    try:
        while loc <= instrlen and matches < maxMatches:
            try:
                preloc = preparseFn(instring, loc)
                nextLoc, tokens = parseFn(instring, preloc, callPreParse=False)
            except ParseException:
                loc = preloc + 1
            else:
                if nextLoc > loc:
                    matches += 1
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
            raise exc