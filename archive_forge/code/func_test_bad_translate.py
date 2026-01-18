from taskflow import deciders
from taskflow import test
def test_bad_translate(self):
    self.assertRaises(TypeError, deciders.Depth.translate, 3)
    self.assertRaises(TypeError, deciders.Depth.translate, object())
    self.assertRaises(ValueError, deciders.Depth.translate, 'stuff')