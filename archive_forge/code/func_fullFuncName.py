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
def fullFuncName(func):
    qualName = str(pickle.whichmodule(func, func.__name__)) + '.' + func.__name__
    if namedObject(qualName) is not func:
        raise Exception(f"Couldn't find {func} as {qualName}.")
    return qualName