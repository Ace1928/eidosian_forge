from io import StringIO
import logging
import unittest
from numba.core import tracing
@unittest.skip('recursive decoration not yet implemented')
def test_injected(self):
    with self.capture:
        tracing.trace(Class2, recursive=True)
        Class2.class_method()
        Class2.static_method()
        test = Class2()
        test.test = 1
        assert 1 == test.test
        test.method()
        self.assertEqual(self.capture.getvalue(), ">> Class2.class_method(cls=<type 'Class2'>)\n" + '<< Class2.class_method\n>> static_method()\n<< static_method\n')