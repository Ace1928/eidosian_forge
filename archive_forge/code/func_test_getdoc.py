from contextlib import contextmanager
from inspect import signature, Signature, Parameter
import inspect
import os
import pytest
import re
import sys
from .. import oinspect
from decorator import decorator
from IPython.testing.tools import AssertPrints, AssertNotPrints
from IPython.utils.path import compress_user
def test_getdoc():

    class A(object):
        """standard docstring"""
        pass

    class B(object):
        """standard docstring"""

        def getdoc(self):
            return 'custom docstring'

    class C(object):
        """standard docstring"""

        def getdoc(self):
            return None
    a = A()
    b = B()
    c = C()
    assert oinspect.getdoc(a) == 'standard docstring'
    assert oinspect.getdoc(b) == 'custom docstring'
    assert oinspect.getdoc(c) == 'standard docstring'