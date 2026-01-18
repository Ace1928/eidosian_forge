from taskflow import deciders
from taskflow import test
def test_bad_pick_widest(self):
    self.assertRaises(ValueError, deciders.pick_widest, [])
    self.assertRaises(ValueError, deciders.pick_widest, ['a'])
    self.assertRaises(ValueError, deciders.pick_widest, set(['b']))