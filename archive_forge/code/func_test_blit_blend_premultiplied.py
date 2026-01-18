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
def test_blit_blend_premultiplied(self):

    def test_premul_surf(src_col, dst_col, src_size=(16, 16), dst_size=(16, 16), src_bit_depth=32, dst_bit_depth=32, src_has_alpha=True, dst_has_alpha=True):
        if src_bit_depth == 8:
            src = pygame.Surface(src_size, 0, src_bit_depth)
            palette = [src_col, dst_col]
            src.set_palette(palette)
            src.fill(palette[0])
        elif src_has_alpha:
            src = pygame.Surface(src_size, SRCALPHA, src_bit_depth)
            src.fill(src_col)
        else:
            src = pygame.Surface(src_size, 0, src_bit_depth)
            src.fill(src_col)
        if dst_bit_depth == 8:
            dst = pygame.Surface(dst_size, 0, dst_bit_depth)
            palette = [src_col, dst_col]
            dst.set_palette(palette)
            dst.fill(palette[1])
        elif dst_has_alpha:
            dst = pygame.Surface(dst_size, SRCALPHA, dst_bit_depth)
            dst.fill(dst_col)
        else:
            dst = pygame.Surface(dst_size, 0, dst_bit_depth)
            dst.fill(dst_col)
        dst.blit(src, (0, 0), special_flags=BLEND_PREMULTIPLIED)
        actual_col = dst.get_at((int(float(src_size[0] / 2.0)), int(float(src_size[0] / 2.0))))
        if src_col.a == 0:
            expected_col = dst_col
        elif src_col.a == 255:
            expected_col = src_col
        else:
            expected_col = pygame.Color(src_col.r + dst_col.r - ((dst_col.r + 1) * src_col.a >> 8), src_col.g + dst_col.g - ((dst_col.g + 1) * src_col.a >> 8), src_col.b + dst_col.b - ((dst_col.b + 1) * src_col.a >> 8), src_col.a + dst_col.a - ((dst_col.a + 1) * src_col.a >> 8))
        if not dst_has_alpha:
            expected_col.a = 255
        return (expected_col, actual_col)
    self.assertEqual(*test_premul_surf(pygame.Color(40, 20, 0, 51), pygame.Color(40, 20, 0, 51)))
    self.assertEqual(*test_premul_surf(pygame.Color(0, 0, 0, 0), pygame.Color(40, 20, 0, 51)))
    self.assertEqual(*test_premul_surf(pygame.Color(40, 20, 0, 51), pygame.Color(0, 0, 0, 0)))
    self.assertEqual(*test_premul_surf(pygame.Color(0, 0, 0, 0), pygame.Color(0, 0, 0, 0)))
    self.assertEqual(*test_premul_surf(pygame.Color(2, 2, 2, 2), pygame.Color(40, 20, 0, 51)))
    self.assertEqual(*test_premul_surf(pygame.Color(40, 20, 0, 51), pygame.Color(2, 2, 2, 2)))
    self.assertEqual(*test_premul_surf(pygame.Color(2, 2, 2, 2), pygame.Color(2, 2, 2, 2)))
    self.assertEqual(*test_premul_surf(pygame.Color(9, 9, 9, 9), pygame.Color(40, 20, 0, 51)))
    self.assertEqual(*test_premul_surf(pygame.Color(40, 20, 0, 51), pygame.Color(9, 9, 9, 9)))
    self.assertEqual(*test_premul_surf(pygame.Color(9, 9, 9, 9), pygame.Color(9, 9, 9, 9)))
    self.assertEqual(*test_premul_surf(pygame.Color(127, 127, 127, 127), pygame.Color(40, 20, 0, 51)))
    self.assertEqual(*test_premul_surf(pygame.Color(40, 20, 0, 51), pygame.Color(127, 127, 127, 127)))
    self.assertEqual(*test_premul_surf(pygame.Color(127, 127, 127, 127), pygame.Color(127, 127, 127, 127)))
    self.assertEqual(*test_premul_surf(pygame.Color(200, 200, 200, 200), pygame.Color(40, 20, 0, 51)))
    self.assertEqual(*test_premul_surf(pygame.Color(40, 20, 0, 51), pygame.Color(200, 200, 200, 200)))
    self.assertEqual(*test_premul_surf(pygame.Color(200, 200, 200, 200), pygame.Color(200, 200, 200, 200)))
    self.assertEqual(*test_premul_surf(pygame.Color(255, 255, 255, 255), pygame.Color(40, 20, 0, 51)))
    self.assertEqual(*test_premul_surf(pygame.Color(40, 20, 0, 51), pygame.Color(255, 255, 255, 255)))
    self.assertEqual(*test_premul_surf(pygame.Color(255, 255, 255, 255), pygame.Color(255, 255, 255, 255)))
    self.assertRaises(IndexError, test_premul_surf, pygame.Color(255, 255, 255, 255), pygame.Color(255, 255, 255, 255), src_size=(0, 0), dst_size=(0, 0))
    self.assertEqual(*test_premul_surf(pygame.Color(40, 20, 0, 51), pygame.Color(30, 20, 0, 51), src_size=(4, 4), dst_size=(9, 9)))
    self.assertEqual(*test_premul_surf(pygame.Color(30, 20, 0, 51), pygame.Color(40, 20, 0, 51), src_size=(17, 67), dst_size=(69, 69)))
    self.assertEqual(*test_premul_surf(pygame.Color(30, 20, 0, 255), pygame.Color(40, 20, 0, 51), src_size=(17, 67), dst_size=(69, 69), src_has_alpha=True))
    self.assertEqual(*test_premul_surf(pygame.Color(30, 20, 0, 51), pygame.Color(40, 20, 0, 255), src_size=(17, 67), dst_size=(69, 69), dst_has_alpha=False))
    self.assertEqual(*test_premul_surf(pygame.Color(30, 20, 0, 255), pygame.Color(40, 20, 0, 255), src_size=(17, 67), dst_size=(69, 69), src_has_alpha=False, dst_has_alpha=False))
    self.assertEqual(*test_premul_surf(pygame.Color(30, 20, 0, 255), pygame.Color(40, 20, 0, 255), src_size=(17, 67), dst_size=(69, 69), dst_bit_depth=24, src_has_alpha=True, dst_has_alpha=False))
    self.assertEqual(*test_premul_surf(pygame.Color(30, 20, 0, 255), pygame.Color(40, 20, 0, 255), src_size=(17, 67), dst_size=(69, 69), src_bit_depth=24, src_has_alpha=False, dst_has_alpha=True))
    self.assertEqual(*test_premul_surf(pygame.Color(30, 20, 0, 255), pygame.Color(40, 20, 0, 255), src_size=(17, 67), dst_size=(69, 69), src_bit_depth=24, dst_bit_depth=24, src_has_alpha=False, dst_has_alpha=False))
    self.assertEqual(*test_premul_surf(pygame.Color(30, 20, 0, 255), pygame.Color(40, 20, 0, 255), src_size=(17, 67), dst_size=(69, 69), src_bit_depth=8))
    self.assertEqual(*test_premul_surf(pygame.Color(30, 20, 0, 255), pygame.Color(40, 20, 0, 255), src_size=(17, 67), dst_size=(69, 69), dst_bit_depth=8))
    self.assertEqual(*test_premul_surf(pygame.Color(30, 20, 0, 255), pygame.Color(40, 20, 0, 255), src_size=(17, 67), dst_size=(69, 69), src_bit_depth=8, dst_bit_depth=8))