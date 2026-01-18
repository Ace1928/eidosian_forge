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
def testSavePNG8(self):
    """see if we can save an 8 bit png correctly"""
    set_pixels = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (170, 146, 170)]
    size = (1, len(set_pixels))
    surf = pygame.Surface(size, depth=8)
    for cnt, pix in enumerate(set_pixels):
        surf.set_at((0, cnt), pix)
    f_path = tempfile.mktemp(suffix='.png')
    pygame.image.save(surf, f_path)
    try:
        reader = png.Reader(filename=f_path)
        width, height, pixels, _ = reader.asRGB8()
        self.assertEqual(size, (width, height))
        self.assertEqual(list(map(tuple, pixels)), set_pixels)
    finally:
        if not reader.file.closed:
            reader.file.close()
        del reader
        os.remove(f_path)