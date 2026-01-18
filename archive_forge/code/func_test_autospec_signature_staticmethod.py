import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
def test_autospec_signature_staticmethod(self):

    class Foo:

        @staticmethod
        def static_method(a, b=10, *, c):
            pass
    mock = create_autospec(Foo.__dict__['static_method'])
    self.assertEqual(inspect.signature(Foo.static_method), inspect.signature(mock))