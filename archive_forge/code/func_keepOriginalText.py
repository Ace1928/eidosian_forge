import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def keepOriginalText(s, startLoc, t):
    """DEPRECATED - use new helper method C{L{originalTextFor}}.
       Helper parse action to preserve original parsed text,
       overriding any nested parse actions."""
    try:
        endloc = getTokensEndLoc()
    except ParseException:
        raise ParseFatalException('incorrect usage of keepOriginalText - may only be called as a parse action')
    del t[:]
    t += ParseResults(s[startLoc:endloc])
    return t