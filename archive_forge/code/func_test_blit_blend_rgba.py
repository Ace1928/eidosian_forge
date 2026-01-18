import os
import unittest
from pygame.tests import test_utils
from pygame.tests.test_utils import (
import pygame
from pygame.locals import *
from pygame.bufferproxy import BufferProxy
import platform
import gc
import weakref
import ctypes
def test_blit_blend_rgba(self):
    sources = [self._make_src_surface(8), self._make_src_surface(16), self._make_src_surface(16, srcalpha=True), self._make_src_surface(24), self._make_src_surface(32), self._make_src_surface(32, srcalpha=True)]
    destinations = [self._make_surface(8), self._make_surface(16), self._make_surface(16, srcalpha=True), self._make_surface(24), self._make_surface(32), self._make_surface(32, srcalpha=True)]
    blend = [('BLEND_RGBA_ADD', (0, 25, 100, 255), lambda a, b: min(a + b, 255)), ('BLEND_RGBA_SUB', (0, 25, 100, 255), lambda a, b: max(a - b, 0)), ('BLEND_RGBA_MULT', (0, 7, 100, 255), lambda a, b: a * b + 255 >> 8), ('BLEND_RGBA_MIN', (0, 255, 0, 255), min), ('BLEND_RGBA_MAX', (0, 255, 0, 255), max)]
    for src in sources:
        src_palette = [src.unmap_rgb(src.map_rgb(c)) for c in self._test_palette]
        for dst in destinations:
            for blend_name, dst_color, op in blend:
                dc = dst.unmap_rgb(dst.map_rgb(dst_color))
                p = []
                for sc in src_palette:
                    c = [op(dc[i], sc[i]) for i in range(4)]
                    if not dst.get_masks()[3]:
                        c[3] = 255
                    c = dst.unmap_rgb(dst.map_rgb(c))
                    p.append(c)
                dst.fill(dst_color)
                dst.blit(src, (0, 0), special_flags=getattr(pygame, blend_name))
                self._assert_surface(dst, p, ', op: %s, src bpp: %i, src flags: %i' % (blend_name, src.get_bitsize(), src.get_flags()))
    src = self._make_src_surface(32, srcalpha=True)
    masks = src.get_masks()
    dst = pygame.Surface(src.get_size(), SRCALPHA, 32, (masks[2], masks[1], masks[0], masks[3]))
    for blend_name, dst_color, op in blend:
        p = [tuple((op(dst_color[i], src_color[i]) for i in range(4))) for src_color in self._test_palette]
        dst.fill(dst_color)
        dst.blit(src, (0, 0), special_flags=getattr(pygame, blend_name))
        self._assert_surface(dst, p, f', {blend_name}')
    src = pygame.Surface((8, 10), SRCALPHA, 32)
    dst = pygame.Surface((8, 10), SRCALPHA, 32)
    tst = pygame.Surface((8, 10), SRCALPHA, 32)
    src.fill((1, 2, 3, 4))
    dst.fill((40, 30, 20, 10))
    subsrc = src.subsurface((2, 3, 4, 4))
    subdst = dst.subsurface((2, 3, 4, 4))
    subdst.blit(subsrc, (0, 0), special_flags=BLEND_RGBA_ADD)
    tst.fill((40, 30, 20, 10))
    tst.fill((41, 32, 23, 14), (2, 3, 4, 4))
    for x in range(8):
        for y in range(10):
            self.assertEqual(dst.get_at((x, y)), tst.get_at((x, y)), '%s != %s at (%i, %i)' % (dst.get_at((x, y)), tst.get_at((x, y)), x, y))