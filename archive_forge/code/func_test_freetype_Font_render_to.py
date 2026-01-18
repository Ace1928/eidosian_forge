import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_render_to(self):
    font = self._TEST_FONTS['sans']
    surf = pygame.Surface((800, 600))
    color = pygame.Color(0, 0, 0)
    rrect = font.render_to(surf, (32, 32), 'FoobarBaz', color, None, size=24)
    self.assertIsInstance(rrect, pygame.Rect)
    self.assertEqual(rrect.topleft, (32, 32))
    self.assertNotEqual(rrect.bottomright, (32, 32))
    rcopy = rrect.copy()
    rcopy.topleft = (32, 32)
    self.assertTrue(surf.get_rect().contains(rcopy))
    rect = pygame.Rect(20, 20, 2, 2)
    rrect = font.render_to(surf, rect, 'FoobarBax', color, None, size=24)
    self.assertEqual(rect.topleft, rrect.topleft)
    self.assertNotEqual(rrect.size, rect.size)
    rrect = font.render_to(surf, (20.1, 18.9), 'FoobarBax', color, None, size=24)
    rrect = font.render_to(surf, rect, '', color, None, size=24)
    self.assertFalse(rrect)
    self.assertEqual(rrect.height, font.get_sized_height(24))
    self.assertRaises(TypeError, font.render_to, 'not a surface', 'text', color)
    self.assertRaises(TypeError, font.render_to, pygame.Surface, 'text', color)
    for dest in [None, 0, 'a', 'ab', (), (1,), ('a', 2), (1, 'a'), (1 + 2j, 2), (1, 1 + 2j), (1, int), (int, 1)]:
        self.assertRaises(TypeError, font.render_to, surf, dest, 'foobar', color, size=24)
    self.assertRaises(ValueError, font.render_to, surf, (0, 0), 'foobar', color)
    self.assertRaises(TypeError, font.render_to, surf, (0, 0), 'foobar', color, 2.3, size=24)
    self.assertRaises(ValueError, font.render_to, surf, (0, 0), 'foobar', color, None, style=42, size=24)
    self.assertRaises(TypeError, font.render_to, surf, (0, 0), 'foobar', color, None, style=None, size=24)
    self.assertRaises(ValueError, font.render_to, surf, (0, 0), 'foobar', color, None, style=97, size=24)