import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def matchOnlyAtCol(n):
    """Helper method for defining parse actions that require matching at a specific
       column in the input text.
    """

    def verifyCol(strg, locn, toks):
        if col(locn, strg) != n:
            raise ParseException(strg, locn, 'matched token not at column %d' % n)
    return verifyCol