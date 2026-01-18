import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def preParse(self, instring, loc):
    preloc = super(LineStart, self).preParse(instring, loc)
    if instring[preloc] == '\n':
        loc += 1
    return loc