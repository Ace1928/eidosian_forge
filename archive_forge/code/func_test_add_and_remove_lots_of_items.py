import sys
from breezy import tests
from breezy.tests import features
def test_add_and_remove_lots_of_items(self):
    obj = self.module.SimpleSet()
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890'
    for i in chars:
        for j in chars:
            k = (i, j)
            obj.add(k)
    num = len(chars) * len(chars)
    self.assertFillState(num, num, 8191, obj)
    for i in chars:
        for j in chars:
            k = (i, j)
            obj.discard(k)
    self.assertFillState(0, obj.fill, 1023, obj)
    self.assertTrue(obj.fill < 1024 / 5)