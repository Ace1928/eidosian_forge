import unittest
import io
import os
import tempfile
from kivy import setupconfig
def test_keep_data(self):
    root = self.root
    texture = root.texture
    self.assertEqual(root._image._data[0].data, None)
    i1 = self.cls(self.image, keep_data=True)
    if not i1._image._data[0].data:
        self.fail('Image has no data even with keep_data = True')