import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_render(self):
    font = self._TEST_FONTS['sans']
    surf = pygame.Surface((800, 600))
    color = pygame.Color(0, 0, 0)
    rend = font.render('FoobarBaz', pygame.Color(0, 0, 0), None, size=24)
    self.assertIsInstance(rend, tuple)
    self.assertEqual(len(rend), 2)
    self.assertIsInstance(rend[0], pygame.Surface)
    self.assertIsInstance(rend[1], pygame.Rect)
    self.assertEqual(rend[0].get_rect().size, rend[1].size)
    s, r = font.render('', pygame.Color(0, 0, 0), None, size=24)
    self.assertEqual(r.width, 0)
    self.assertEqual(r.height, font.get_sized_height(24))
    self.assertEqual(s.get_size(), r.size)
    self.assertEqual(s.get_bitsize(), 32)
    self.assertRaises(ValueError, font.render, 'foobar', color)
    self.assertRaises(TypeError, font.render, 'foobar', color, 2.3, size=24)
    self.assertRaises(ValueError, font.render, 'foobar', color, None, style=42, size=24)
    self.assertRaises(TypeError, font.render, 'foobar', color, None, style=None, size=24)
    self.assertRaises(ValueError, font.render, 'foobar', color, None, style=97, size=24)
    font2 = self._TEST_FONTS['mono']
    ucs4 = font2.ucs4
    try:
        font2.ucs4 = False
        rend1 = font2.render('\ud80c\udc79', color, size=24)
        rend2 = font2.render('𓁹', color, size=24)
        self.assertEqual(rend1[1], rend2[1])
        font2.ucs4 = True
        rend1 = font2.render('\ud80c\udc79', color, size=24)
        self.assertNotEqual(rend1[1], rend2[1])
    finally:
        font2.ucs4 = ucs4
    self.assertRaises(UnicodeEncodeError, font.render, '\ud80c', color, size=24)
    self.assertRaises(UnicodeEncodeError, font.render, '\udca7', color, size=24)
    self.assertRaises(UnicodeEncodeError, font.render, '\ud7ff\udca7', color, size=24)
    self.assertRaises(UnicodeEncodeError, font.render, '\udc00\udca7', color, size=24)
    self.assertRaises(UnicodeEncodeError, font.render, '\ud80c\udbff', color, size=24)
    self.assertRaises(UnicodeEncodeError, font.render, '\ud80c\ue000', color, size=24)
    self.assertRaises(RuntimeError, nullfont().render, 'a', (0, 0, 0), size=24)
    path = os.path.join(FONTDIR, 'A_PyGameMono-8.png')
    A = pygame.image.load(path)
    path = os.path.join(FONTDIR, 'u13079_PyGameMono-8.png')
    u13079 = pygame.image.load(path)
    font = self._TEST_FONTS['mono']
    font.ucs4 = False
    A_rendered, r = font.render('A', bgcolor=pygame.Color('white'), size=8)
    u13079_rendered, r = font.render('𓁹', bgcolor=pygame.Color('white'), size=8)
    bitmap = pygame.Surface(A.get_size(), pygame.SRCALPHA, 32)
    bitmap.blit(A, (0, 0))
    rendering = pygame.Surface(A_rendered.get_size(), pygame.SRCALPHA, 32)
    rendering.blit(A_rendered, (0, 0))
    self.assertTrue(surf_same_image(rendering, bitmap))
    bitmap = pygame.Surface(u13079.get_size(), pygame.SRCALPHA, 32)
    bitmap.blit(u13079, (0, 0))
    rendering = pygame.Surface(u13079_rendered.get_size(), pygame.SRCALPHA, 32)
    rendering.blit(u13079_rendered, (0, 0))
    self.assertTrue(surf_same_image(rendering, bitmap))