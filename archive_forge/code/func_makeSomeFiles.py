import inspect
import sys
import types
import warnings
from os.path import normcase
from warnings import catch_warnings, simplefilter
from incremental import Version
from twisted.python import deprecate
from twisted.python.deprecate import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.python.test import deprecatedattributes
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SynchronousTestCase
from twisted.python.deprecate import deprecatedModuleAttribute
from incremental import Version
from twisted.python import deprecate
from twisted.python import deprecate
def makeSomeFiles(pathobj, dirdict):
    pathdict = {}
    for key, value in dirdict.items():
        child = pathobj.child(key)
        if isinstance(value, bytes):
            pathdict[key] = child
            child.setContent(value)
        elif isinstance(value, dict):
            child.createDirectory()
            pathdict[key] = makeSomeFiles(child, value)
        else:
            raise ValueError('only strings and dicts allowed as values')
    return pathdict