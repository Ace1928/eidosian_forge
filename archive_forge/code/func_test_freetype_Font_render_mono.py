import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_render_mono(self):
    font = self._TEST_FONTS['sans']
    color = pygame.Color('black')
    colorkey = pygame.Color('white')
    text = '.'
    save_antialiased = font.antialiased
    font.antialiased = False
    try:
        surf, r = font.render(text, color, size=24)
        self.assertEqual(surf.get_bitsize(), 8)
        flags = surf.get_flags()
        self.assertTrue(flags & pygame.SRCCOLORKEY)
        self.assertFalse(flags & (pygame.SRCALPHA | pygame.HWSURFACE))
        self.assertEqual(surf.get_colorkey(), colorkey)
        self.assertIsNone(surf.get_alpha())
        translucent_color = pygame.Color(*color)
        translucent_color.a = 55
        surf, r = font.render(text, translucent_color, size=24)
        self.assertEqual(surf.get_bitsize(), 8)
        flags = surf.get_flags()
        self.assertTrue(flags & (pygame.SRCCOLORKEY | pygame.SRCALPHA))
        self.assertFalse(flags & pygame.HWSURFACE)
        self.assertEqual(surf.get_colorkey(), colorkey)
        self.assertEqual(surf.get_alpha(), translucent_color.a)
        surf, r = font.render(text, color, colorkey, size=24)
        self.assertEqual(surf.get_bitsize(), 32)
    finally:
        font.antialiased = save_antialiased