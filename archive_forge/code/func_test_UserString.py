import array
import unittest
from collections import OrderedDict
from collections import abc
from collections import deque
from types import MappingProxyType
from zope.interface import Invalid
from zope.interface._compat import PYPY
from zope.interface.common import collections
from . import VerifyClassMixin
from . import VerifyObjectMixin
from . import add_abc_interface_tests
def test_UserString(self):
    self.assertTrue(self.verify(collections.ISequence, collections.UserString))