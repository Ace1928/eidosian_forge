import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_path(self):
    self.assertEqual(self._TEST_FONTS['sans'].path, self._sans_path)
    self.assertRaises(AttributeError, getattr, nullfont(), 'path')