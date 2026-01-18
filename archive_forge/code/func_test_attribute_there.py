import sys
import types
from testtools import TestCase
from testtools.matchers import (
from extras import (
def test_attribute_there(self):

    class Foo(object):
        pass
    foo = Foo()
    foo.attribute = None
    self.assertEqual(True, safe_hasattr(foo, 'attribute'))