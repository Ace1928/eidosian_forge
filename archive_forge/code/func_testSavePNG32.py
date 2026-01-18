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
def testSavePNG32(self):
    """see if we can save a png with color values in the proper channels."""
    reddish_pixel = (215, 0, 0, 255)
    greenish_pixel = (0, 225, 0, 255)
    bluish_pixel = (0, 0, 235, 255)
    greyish_pixel = (115, 125, 135, 145)
    surf = pygame.Surface((1, 4), pygame.SRCALPHA, 32)
    surf.set_at((0, 0), reddish_pixel)
    surf.set_at((0, 1), greenish_pixel)
    surf.set_at((0, 2), bluish_pixel)
    surf.set_at((0, 3), greyish_pixel)
    f_path = tempfile.mktemp(suffix='.png')
    pygame.image.save(surf, f_path)
    try:
        reader = png.Reader(filename=f_path)
        width, height, pixels, metadata = reader.asRGBA8()
        self.assertEqual(tuple(next(pixels)), reddish_pixel)
        self.assertEqual(tuple(next(pixels)), greenish_pixel)
        self.assertEqual(tuple(next(pixels)), bluish_pixel)
        self.assertEqual(tuple(next(pixels)), greyish_pixel)
    finally:
        if not reader.file.closed:
            reader.file.close()
        del reader
        os.remove(f_path)