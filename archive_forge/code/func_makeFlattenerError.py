from __future__ import annotations
import re
import sys
import traceback
from twisted.python.compat import nativeString
from twisted.trial import unittest
from twisted.web import error
from twisted.web.template import Tag
def makeFlattenerError(self, roots: list[object]=[]) -> error.FlattenerError:
    try:
        raise RuntimeError('oh noes')
    except Exception as e:
        tb = traceback.extract_tb(sys.exc_info()[2])
        return error.FlattenerError(e, roots, tb)