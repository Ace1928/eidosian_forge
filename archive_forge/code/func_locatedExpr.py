import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import collections
import pprint
import traceback
import types
from datetime import datetime
def locatedExpr(expr):
    """
    Helper to decorate a returned token with its starting and ending locations in the input string.
    This helper adds the following results names:
     - locn_start = location where matched expression begins
     - locn_end = location where matched expression ends
     - value = the actual parsed results

    Be careful if the input text contains C{<TAB>} characters, you may want to call
    C{L{ParserElement.parseWithTabs}}

    Example::
        wd = Word(alphas)
        for match in locatedExpr(wd).searchString("ljsdf123lksdjjf123lkkjj1222"):
            print(match)
    prints::
        [[0, 'ljsdf', 5]]
        [[8, 'lksdjjf', 15]]
        [[18, 'lkkjj', 23]]
    """
    locator = Empty().setParseAction(lambda s, l, t: l)
    return Group(locator('locn_start') + expr('value') + locator.copy().leaveWhitespace()('locn_end'))