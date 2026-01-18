import os
import re
import sys
import unittest
from collections import defaultdict
from kivy.core.image import ImageLoader
def test_internal_converter_2x1(self):
    correct = {'rgba': b'\x01\x02\x03\xa1\x04\x05\x06\xa2', 'abgr': b'\xa1\x03\x02\x01\xa2\x06\x05\x04', 'bgra': b'\x03\x02\x01\xa1\x06\x05\x04\xa2', 'argb': b'\xa1\x01\x02\x03\xa2\x04\x05\x06', 'rgb': b'\x01\x02\x03\x04\x05\x06', 'bgr': b'\x03\x02\x01\x06\x05\x04', 'rgb_align4': b'\x01\x02\x03\x04\x05\x06\x00\x00', 'bgr_align4': b'\x03\x02\x01\x06\x05\x04\x00\x00'}
    src = correct.get
    rgba = src('rgba')
    self.assertEqual(rgba_to(rgba, 'rgba', 2, 1, 0), src('rgba'))
    self.assertEqual(rgba_to(rgba, 'abgr', 2, 1, 0), src('abgr'))
    self.assertEqual(rgba_to(rgba, 'bgra', 2, 1, 0), src('bgra'))
    self.assertEqual(rgba_to(rgba, 'argb', 2, 1, 0), src('argb'))
    self.assertEqual(rgba_to(rgba, 'rgb', 2, 1, 0), src('rgb'))
    self.assertEqual(rgba_to(rgba, 'bgr', 2, 1, 0), src('bgr'))
    self.assertEqual(rgba_to(rgba, 'rgb', 2, 1, None), src('rgb_align4'))
    self.assertEqual(rgba_to(rgba, 'bgr', 2, 1, None), src('bgr_align4'))