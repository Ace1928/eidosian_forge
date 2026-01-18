import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def searchString(self, instring, maxMatches=_MAX_INT):
    """Another extension to C{L{scanString}}, simplifying the access to the tokens found
           to match the given parse expression.  May be called with optional
           C{maxMatches} argument, to clip searching after 'n' matches are found.
        """
    try:
        return ParseResults([t for t, s, e in self.scanString(instring, maxMatches)])
    except ParseBaseException as exc:
        if ParserElement.verbose_stacktrace:
            raise
        else:
            raise exc