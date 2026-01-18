import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def inlineLiteralsUsing(cls):
    """
        Set class to be used for inclusion of string literals into a parser.
        """
    ParserElement.literalStringClass = cls