from doctest import ELLIPSIS
from pprint import pformat
import sys
import _thread
import unittest
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.matchers import (
from testtools.testcase import (
from testtools.testresult.doubles import (
from testtools.tests.helpers import (
from testtools.tests.samplecases import (
def test_called_with_arguments(self):
    l = []

    def foo(*args, **kwargs):
        l.append((args, kwargs))
    wrapped = Nullary(foo, 1, 2, a='b')
    wrapped()
    self.assertEqual(l, [((1, 2), {'a': 'b'})])