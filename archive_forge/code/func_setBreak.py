import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def setBreak(self, breakFlag=True):
    """Method to invoke the Python pdb debugger when this element is
           about to be parsed. Set C{breakFlag} to True to enable, False to
           disable.
        """
    if breakFlag:
        _parseMethod = self._parse

        def breaker(instring, loc, doActions=True, callPreParse=True):
            import pdb
            pdb.set_trace()
            return _parseMethod(instring, loc, doActions, callPreParse)
        breaker._originalParseMethod = _parseMethod
        self._parse = breaker
    elif hasattr(self._parse, '_originalParseMethod'):
        self._parse = self._parse._originalParseMethod
    return self