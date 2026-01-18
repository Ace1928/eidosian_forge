import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def setDefaultWhitespaceChars(chars):
    """Overrides the default whitespace chars
        """
    ParserElement.DEFAULT_WHITE_CHARS = chars