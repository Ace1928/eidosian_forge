import os
import sys
from collections import OrderedDict
import unittest
from unittest.test.testmock import support
from unittest.test.testmock.support import SomeClass, is_instance
from test.test_importlib.util import uncache
from unittest.mock import (
def test_autospec_classmethod_signature(self):

    class Foo:

        @classmethod
        def class_method(cls, a, b=10, *, c):
            pass
    Foo.class_method(1, 2, c=3)
    with patch.object(Foo, 'class_method', autospec=True) as method:
        method(1, 2, c=3)
        self.assertRaises(TypeError, method)
        self.assertRaises(TypeError, method, 1)
        self.assertRaises(TypeError, method, 1, 2, 3, c=4)