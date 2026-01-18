import array
import binascii
import io
import os
import tempfile
import unittest
import glob
import pathlib
from pygame.tests.test_utils import example_path, png, tostring
import pygame, pygame.image, pygame.pkgdata
@unittest.skipIf(sdl_image_svg_jpeg_save_bug, 'SDL_image 2.0.5 and older has a big endian bug in jpeg saving')
def testSaveJPG(self):
    """JPG equivalent to issue #211 - color channel swapping

        Make sure the SDL surface color masks represent the rgb memory format
        required by the JPG library. The masks are machine endian dependent
        """
    from pygame import Color, Rect
    square_len = 16
    sz = (2 * square_len, 2 * square_len)

    def as_rect(square_x, square_y):
        return Rect(square_x * square_len, square_y * square_len, square_len, square_len)
    squares = [(as_rect(0, 0), Color('red')), (as_rect(1, 0), Color('green')), (as_rect(0, 1), Color('blue')), (as_rect(1, 1), Color(255, 128, 64))]
    surf = pygame.Surface(sz, 0, 32)
    for rect, color in squares:
        surf.fill(color, rect)
    f_path = tempfile.mktemp(suffix='.jpg')
    pygame.image.save(surf, f_path)
    jpg_surf = pygame.image.load(f_path)

    def approx(c):
        mask = 252
        return pygame.Color(c.r & mask, c.g & mask, c.b & mask)
    offset = square_len // 2
    for rect, color in squares:
        posn = rect.move((offset, offset)).topleft
        self.assertEqual(approx(jpg_surf.get_at(posn)), approx(color))
    os.remove(f_path)