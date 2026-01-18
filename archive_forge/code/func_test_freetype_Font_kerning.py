import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_kerning(self):
    """Ensures get/set works with the kerning property."""
    ft_font = self._TEST_FONTS['sans']
    self.assertFalse(ft_font.kerning)
    ft_font.kerning = True
    self.assertTrue(ft_font.kerning)
    ft_font.kerning = False
    self.assertFalse(ft_font.kerning)