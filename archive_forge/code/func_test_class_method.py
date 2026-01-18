from io import StringIO
import logging
import unittest
from numba.core import tracing
def test_class_method(self):
    with self.capture:
        Class.class_method()
    self.assertEqual(self.capture.getvalue(), ">> Class.class_method(cls=<class 'Class'>)\n" + '<< Class.class_method\n')