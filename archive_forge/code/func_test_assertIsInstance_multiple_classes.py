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
def test_assertIsInstance_multiple_classes(self):

    class Foo:
        """Simple class for testing assertIsInstance."""

    class Bar:
        """Another simple class for testing assertIsInstance."""
    foo = Foo()
    self.assertIsInstance(foo, (Foo, Bar))
    self.assertIsInstance(Bar(), (Foo, Bar))