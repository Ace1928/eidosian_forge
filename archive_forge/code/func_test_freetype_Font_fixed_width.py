import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_fixed_width(self):
    f = self._TEST_FONTS['sans']
    self.assertFalse(f.fixed_width)
    f = self._TEST_FONTS['mono']
    self.assertTrue(f.fixed_width)
    self.assertRaises(RuntimeError, lambda: nullfont().fixed_width)