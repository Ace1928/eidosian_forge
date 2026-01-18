import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def originalTextFor(expr, asString=True):
    """Helper to return the original, untokenized text for a given expression.  Useful to
       restore the parsed fields of an HTML start tag into the raw tag text itself, or to
       revert separate tokens with intervening whitespace back to the original matching
       input text. Simpler to use than the parse action C{L{keepOriginalText}}, and does not
       require the inspect module to chase up the call stack.  By default, returns a 
       string containing the original parsed text.  
       
       If the optional C{asString} argument is passed as C{False}, then the return value is a 
       C{L{ParseResults}} containing any results names that were originally matched, and a 
       single token containing the original matched text from the input string.  So if 
       the expression passed to C{L{originalTextFor}} contains expressions with defined
       results names, you must set C{asString} to C{False} if you want to preserve those
       results name values."""
    locMarker = Empty().setParseAction(lambda s, loc, t: loc)
    endlocMarker = locMarker.copy()
    endlocMarker.callPreparse = False
    matchExpr = locMarker('_original_start') + expr + endlocMarker('_original_end')
    if asString:
        extractText = lambda s, l, t: s[t._original_start:t._original_end]
    else:

        def extractText(s, l, t):
            del t[:]
            t.insert(0, s[t._original_start:t._original_end])
            del t['_original_start']
            del t['_original_end']
    matchExpr.setParseAction(extractText)
    return matchExpr