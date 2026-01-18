import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def nullDebugAction(*args):
    """'Do-nothing' debug action, to suppress debugging output during parsing."""
    pass