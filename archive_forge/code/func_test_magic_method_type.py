from __future__ import division
import inspect
import sys
import textwrap
import six
import unittest2 as unittest
from mock import Mock, MagicMock
from mock.mock import _magics
def test_magic_method_type(self):

    class Foo(MagicMock):
        pass
    foo = Foo()
    self.assertIsInstance(foo.__int__, Foo)