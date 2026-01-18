import pickle
import unittest
from pyomo.common.collections import Bunch
def test_item_methods(self):
    b = Bunch()
    b['foo'] = 5
    self.assertEqual(list(b.keys()), ['foo'])
    self.assertEqual(b['foo'], 5)
    b['_foo'] = 50
    self.assertEqual(list(b.keys()), ['foo'])
    self.assertEqual(b['foo'], 5)
    self.assertEqual(b['_foo'], 50)
    del b['foo']
    self.assertEqual(list(b.keys()), [])
    self.assertEqual(b['foo'], None)
    self.assertEqual(b['_foo'], 50)
    del b['_foo']
    self.assertEqual(list(b.keys()), [])
    self.assertEqual(b['foo'], None)
    with self.assertRaisesRegex(AttributeError, "Unknown attribute '_foo'"):
        b['_foo']
    with self.assertRaisesRegex(ValueError, 'Bunch keys must be str \\(got int\\)'):
        b[5]
    with self.assertRaisesRegex(ValueError, 'Bunch keys must be str \\(got int\\)'):
        b[5] = 5
    with self.assertRaisesRegex(ValueError, 'Bunch keys must be str \\(got int\\)'):
        del b[5]