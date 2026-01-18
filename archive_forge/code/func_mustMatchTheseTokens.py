import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def mustMatchTheseTokens(s, l, t):
    theseTokens = _flatten(t.asList())
    if theseTokens != matchTokens:
        raise ParseException('', 0, '')