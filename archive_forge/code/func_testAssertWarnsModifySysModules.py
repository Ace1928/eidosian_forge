import contextlib
import difflib
import pprint
import pickle
import re
import sys
import logging
import warnings
import weakref
import inspect
import types
from copy import deepcopy
from test import support
import unittest
from unittest.test.support import (
from test.support import captured_stderr, gc_collect
def testAssertWarnsModifySysModules(self):

    class Foo(types.ModuleType):

        @property
        def __warningregistry__(self):
            sys.modules['@bar@'] = 'bar'
    sys.modules['@foo@'] = Foo('foo')
    try:
        self.assertWarns(UserWarning, warnings.warn, 'expected')
    finally:
        del sys.modules['@foo@']
        del sys.modules['@bar@']