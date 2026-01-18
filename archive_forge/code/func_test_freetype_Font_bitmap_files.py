import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_bitmap_files(self):
    """Ensure bitmap file restrictions are caught"""
    f = self._TEST_FONTS['bmp-8-75dpi']
    f_null = nullfont()
    s = pygame.Surface((10, 10), 0, 32)
    a = s.get_view('3')
    exception = AttributeError
    self.assertRaises(exception, setattr, f, 'strong', True)
    self.assertRaises(exception, setattr, f, 'oblique', True)
    self.assertRaises(exception, setattr, f, 'style', ft.STYLE_STRONG)
    self.assertRaises(exception, setattr, f, 'style', ft.STYLE_OBLIQUE)
    exception = RuntimeError
    self.assertRaises(exception, setattr, f_null, 'strong', True)
    self.assertRaises(exception, setattr, f_null, 'oblique', True)
    self.assertRaises(exception, setattr, f_null, 'style', ft.STYLE_STRONG)
    self.assertRaises(exception, setattr, f_null, 'style', ft.STYLE_OBLIQUE)
    exception = ValueError
    self.assertRaises(exception, f.render, 'A', (0, 0, 0), size=8, rotation=1)
    self.assertRaises(exception, f.render, 'A', (0, 0, 0), size=8, style=ft.STYLE_OBLIQUE)
    self.assertRaises(exception, f.render, 'A', (0, 0, 0), size=8, style=ft.STYLE_STRONG)
    self.assertRaises(exception, f.render_raw, 'A', size=8, rotation=1)
    self.assertRaises(exception, f.render_raw, 'A', size=8, style=ft.STYLE_OBLIQUE)
    self.assertRaises(exception, f.render_raw, 'A', size=8, style=ft.STYLE_STRONG)
    self.assertRaises(exception, f.render_to, s, (0, 0), 'A', (0, 0, 0), size=8, rotation=1)
    self.assertRaises(exception, f.render_to, s, (0, 0), 'A', (0, 0, 0), size=8, style=ft.STYLE_OBLIQUE)
    self.assertRaises(exception, f.render_to, s, (0, 0), 'A', (0, 0, 0), size=8, style=ft.STYLE_STRONG)
    self.assertRaises(exception, f.render_raw_to, a, 'A', size=8, rotation=1)
    self.assertRaises(exception, f.render_raw_to, a, 'A', size=8, style=ft.STYLE_OBLIQUE)
    self.assertRaises(exception, f.render_raw_to, a, 'A', size=8, style=ft.STYLE_STRONG)
    self.assertRaises(exception, f.get_rect, 'A', size=8, rotation=1)
    self.assertRaises(exception, f.get_rect, 'A', size=8, style=ft.STYLE_OBLIQUE)
    self.assertRaises(exception, f.get_rect, 'A', size=8, style=ft.STYLE_STRONG)
    exception = pygame.error
    self.assertRaises(exception, f.get_rect, 'A', size=42)
    self.assertRaises(exception, f.get_metrics, 'A', size=42)
    self.assertRaises(exception, f.get_sized_ascender, 42)
    self.assertRaises(exception, f.get_sized_descender, 42)
    self.assertRaises(exception, f.get_sized_height, 42)
    self.assertRaises(exception, f.get_sized_glyph_height, 42)