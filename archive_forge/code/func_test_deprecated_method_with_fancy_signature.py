import unittest
from traits.testing.api import UnittestTools
from traits.util.api import deprecated
def test_deprecated_method_with_fancy_signature(self):
    obj = ClassWithDeprecatedBits()
    with self.assertDeprecated():
        result = obj.bytes(3, 27, 65, name='Boris', age=-3.2)
    self.assertEqual(result, (3, (27, 65), {'name': 'Boris', 'age': -3.2}))