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
def test_blit_blend(self):
    sources = [self._make_src_surface(8), self._make_src_surface(16), self._make_src_surface(16, srcalpha=True), self._make_src_surface(24), self._make_src_surface(32), self._make_src_surface(32, srcalpha=True)]
    destinations = [self._make_surface(8), self._make_surface(16), self._make_surface(16, srcalpha=True), self._make_surface(24), self._make_surface(32), self._make_surface(32, srcalpha=True)]
    blend = [('BLEND_ADD', (0, 25, 100, 255), lambda a, b: min(a + b, 255)), ('BLEND_SUB', (100, 25, 0, 100), lambda a, b: max(a - b, 0)), ('BLEND_MULT', (100, 200, 0, 0), lambda a, b: a * b + 255 >> 8), ('BLEND_MIN', (255, 0, 0, 255), min), ('BLEND_MAX', (0, 255, 0, 255), max)]
    for src in sources:
        src_palette = [src.unmap_rgb(src.map_rgb(c)) for c in self._test_palette]
        for dst in destinations:
            for blend_name, dst_color, op in blend:
                dc = dst.unmap_rgb(dst.map_rgb(dst_color))
                p = []
                for sc in src_palette:
                    c = [op(dc[i], sc[i]) for i in range(3)]
                    if dst.get_masks()[3]:
                        c.append(dc[3])
                    else:
                        c.append(255)
                    c = dst.unmap_rgb(dst.map_rgb(c))
                    p.append(c)
                dst.fill(dst_color)
                dst.blit(src, (0, 0), special_flags=getattr(pygame, blend_name))
                self._assert_surface(dst, p, ', op: %s, src bpp: %i, src flags: %i' % (blend_name, src.get_bitsize(), src.get_flags()))
    src = self._make_src_surface(32)
    masks = src.get_masks()
    dst = pygame.Surface(src.get_size(), 0, 32, [masks[2], masks[1], masks[0], masks[3]])
    for blend_name, dst_color, op in blend:
        p = []
        for src_color in self._test_palette:
            c = [op(dst_color[i], src_color[i]) for i in range(3)]
            c.append(255)
            p.append(tuple(c))
        dst.fill(dst_color)
        dst.blit(src, (0, 0), special_flags=getattr(pygame, blend_name))
        self._assert_surface(dst, p, f', {blend_name}')
    pat = self._make_src_surface(32)
    masks = pat.get_masks()
    if min(masks) == 4278190080:
        masks = [m >> 8 for m in masks]
    else:
        masks = [m << 8 for m in masks]
    src = pygame.Surface(pat.get_size(), 0, 32, masks)
    self._fill_surface(src)
    dst = pygame.Surface(src.get_size(), 0, 32, masks)
    for blend_name, dst_color, op in blend:
        p = []
        for src_color in self._test_palette:
            c = [op(dst_color[i], src_color[i]) for i in range(3)]
            c.append(255)
            p.append(tuple(c))
        dst.fill(dst_color)
        dst.blit(src, (0, 0), special_flags=getattr(pygame, blend_name))
        self._assert_surface(dst, p, f', {blend_name}')