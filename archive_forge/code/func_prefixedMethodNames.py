import os
import pickle
import re
import sys
import traceback
import types
import weakref
from collections import deque
from io import IOBase, StringIO
from typing import Type, Union
from twisted.python.compat import nativeString
from twisted.python.deprecate import _fullyQualifiedName as fullyQualifiedName
def prefixedMethodNames(classObj, prefix):
    """
    Given a class object C{classObj}, returns a list of method names that match
    the string C{prefix}.

    @param classObj: A class object from which to collect method names.

    @param prefix: A native string giving a prefix.  Each method with a name
        which begins with this prefix will be returned.
    @type prefix: L{str}

    @return: A list of the names of matching methods of C{classObj} (and base
        classes of C{classObj}).
    @rtype: L{list} of L{str}
    """
    dct = {}
    addMethodNamesToDict(classObj, dct, prefix)
    return list(dct.keys())