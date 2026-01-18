import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_name(self):
    f = self._TEST_FONTS['sans']
    self.assertEqual(f.name, 'Liberation Sans')
    f = self._TEST_FONTS['fixed']
    self.assertEqual(f.name, 'Inconsolata')
    nf = nullfont()
    self.assertEqual(nf.name, repr(nf))