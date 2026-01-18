import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def asList(self):
    """Returns the parse results as a nested list of matching tokens, all converted to strings."""
    out = []
    for res in self.__toklist:
        if isinstance(res, ParseResults):
            out.append(res.asList())
        else:
            out.append(res)
    return out