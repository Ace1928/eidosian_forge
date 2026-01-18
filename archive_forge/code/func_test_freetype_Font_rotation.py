import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_rotation(self):
    test_angles = [(30, 30), (360, 0), (390, 30), (720, 0), (764, 44), (-30, 330), (-360, 0), (-390, 330), (-720, 0), (-764, 316)]
    f = ft.Font(None)
    self.assertEqual(f.rotation, 0)
    for r, r_reduced in test_angles:
        f.rotation = r
        self.assertEqual(f.rotation, r_reduced, 'for angle %d: %d != %d' % (r, f.rotation, r_reduced))
    self.assertRaises(TypeError, setattr, f, 'rotation', '12')