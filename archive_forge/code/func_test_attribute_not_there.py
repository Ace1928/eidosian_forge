import sys
import types
from testtools import TestCase
from testtools.matchers import (
from extras import (
def test_attribute_not_there(self):

    class Foo(object):
        pass
    self.assertEqual(False, safe_hasattr(Foo(), 'anything'))