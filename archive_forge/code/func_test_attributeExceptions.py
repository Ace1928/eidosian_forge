import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_attributeExceptions(self):
    """
        If segments on the end of a fully-qualified Python name represents
        attributes which aren't actually present on the object represented by
        the earlier segments, L{namedAny} should raise an L{AttributeError}.
        """
    self.assertRaises(AttributeError, reflect.namedAny, 'twisted.nosuchmoduleintheworld')
    self.assertRaises(AttributeError, reflect.namedAny, 'twisted.nosuch.modulein.theworld')
    self.assertRaises(AttributeError, reflect.namedAny, 'twisted.test.test_reflect.Summer.nosuchattribute')