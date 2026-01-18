import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_init(self):
    self.assertRaises(FileNotFoundError, ft.Font, os.path.join(FONTDIR, 'nonexistent.ttf'))
    f = self._TEST_FONTS['sans']
    self.assertIsInstance(f, ft.Font)
    f = self._TEST_FONTS['fixed']
    self.assertIsInstance(f, ft.Font)
    f = ft.Font(size=22, file=None)
    self.assertEqual(f.size, 22)
    f = ft.Font(font_index=0, file=None)
    self.assertNotEqual(ft.get_default_resolution(), 100)
    f = ft.Font(resolution=100, file=None)
    self.assertEqual(f.resolution, 100)
    f = ft.Font(ucs4=True, file=None)
    self.assertTrue(f.ucs4)
    self.assertRaises(OverflowError, ft.Font, file=None, size=max_point_size + 1)
    self.assertRaises(OverflowError, ft.Font, file=None, size=-1)
    f = ft.Font(None, size=24)
    self.assertTrue(f.height > 0)
    self.assertRaises(FileNotFoundError, f.__init__, os.path.join(FONTDIR, 'nonexistent.ttf'))
    f = ft.Font(self._sans_path, size=24, ucs4=True)
    self.assertEqual(f.name, 'Liberation Sans')
    self.assertTrue(f.scalable)
    self.assertFalse(f.fixed_width)
    self.assertTrue(f.antialiased)
    self.assertFalse(f.oblique)
    self.assertTrue(f.ucs4)
    f.antialiased = False
    f.oblique = True
    f.__init__(self._mono_path)
    self.assertEqual(f.name, 'PyGameMono')
    self.assertTrue(f.scalable)
    self.assertTrue(f.fixed_width)
    self.assertFalse(f.antialiased)
    self.assertTrue(f.oblique)
    self.assertTrue(f.ucs4)
    f = ft.Font(self._bmp_8_75dpi_path)
    sizes = f.get_sizes()
    self.assertEqual(len(sizes), 1)
    size_pt, width_px, height_px, x_ppem, y_ppem = sizes[0]
    self.assertEqual(f.size, (x_ppem, y_ppem))
    f.__init__(self._bmp_8_75dpi_path, size=12)
    self.assertEqual(f.size, 12.0)