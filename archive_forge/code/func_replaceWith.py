import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def replaceWith(replStr):
    """Helper method for common parse actions that simply return a literal value.  Especially
       useful when used with C{L{transformString<ParserElement.transformString>}()}.
    """

    def _replFunc(*args):
        return [replStr]
    return _replFunc