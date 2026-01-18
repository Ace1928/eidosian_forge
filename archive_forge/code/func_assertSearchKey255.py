from ... import tests
from .. import chk_map
from ..static_tuple import StaticTuple
def assertSearchKey255(self, expected, key):
    actual = self.module._search_key_255(key)
    self.assertEqual(expected, actual, 'actual: {!r}'.format(actual))