import asyncio
import contextvars
import unittest
from test import support
def test_deprecation_of_return_val_from_test(self):

    class Nothing:

        def __eq__(self, o):
            return o is None

    class Test(unittest.IsolatedAsyncioTestCase):

        async def test1(self):
            return 1

        async def test2(self):
            yield 1

        async def test3(self):
            return Nothing()
    with self.assertWarns(DeprecationWarning) as w:
        Test('test1').run()
    self.assertIn('It is deprecated to return a value that is not None', str(w.warning))
    self.assertIn('test1', str(w.warning))
    self.assertEqual(w.filename, __file__)
    with self.assertWarns(DeprecationWarning) as w:
        Test('test2').run()
    self.assertIn('It is deprecated to return a value that is not None', str(w.warning))
    self.assertIn('test2', str(w.warning))
    self.assertEqual(w.filename, __file__)
    with self.assertWarns(DeprecationWarning) as w:
        Test('test3').run()
    self.assertIn('It is deprecated to return a value that is not None', str(w.warning))
    self.assertIn('test3', str(w.warning))
    self.assertEqual(w.filename, __file__)