import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_text_is_None_with_arr(self):
    f = ft.Font(self._sans_path, 36)
    f.style = ft.STYLE_NORMAL
    f.rotation = 0
    text = 'ABCD'
    get_rect = f.get_rect(text)
    f.vertical = True
    get_rect_vert = f.get_rect(text)
    self.assertTrue(get_rect_vert.width < get_rect.width)
    self.assertTrue(get_rect_vert.height > get_rect.height)
    f.vertical = False
    render_to_surf = pygame.Surface(get_rect.size, pygame.SRCALPHA, 32)
    if IS_PYPY:
        return
    arr = arrinter.Array(get_rect.size, 'u', 1)
    render = f.render(text, (0, 0, 0))
    render_to = f.render_to(render_to_surf, (0, 0), text, (0, 0, 0))
    render_raw = f.render_raw(text)
    render_raw_to = f.render_raw_to(arr, text)
    surf = pygame.Surface(get_rect.size, pygame.SRCALPHA, 32)
    self.assertEqual(f.get_rect(None), get_rect)
    s, r = f.render(None, (0, 0, 0))
    self.assertEqual(r, render[1])
    self.assertTrue(surf_same_image(s, render[0]))
    r = f.render_to(surf, (0, 0), None, (0, 0, 0))
    self.assertEqual(r, render_to)
    self.assertTrue(surf_same_image(surf, render_to_surf))
    px, sz = f.render_raw(None)
    self.assertEqual(sz, render_raw[1])
    self.assertEqual(px, render_raw[0])
    sz = f.render_raw_to(arr, None)
    self.assertEqual(sz, render_raw_to)