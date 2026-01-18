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
def testSavePaletteAsPNG8(self):
    """see if we can save a png with color values in the proper channels."""
    pygame.display.init()
    reddish_pixel = (215, 0, 0)
    greenish_pixel = (0, 225, 0)
    bluish_pixel = (0, 0, 235)
    greyish_pixel = (115, 125, 135)
    surf = pygame.Surface((1, 4), 0, 8)
    surf.set_palette_at(0, reddish_pixel)
    surf.set_palette_at(1, greenish_pixel)
    surf.set_palette_at(2, bluish_pixel)
    surf.set_palette_at(3, greyish_pixel)
    f_path = tempfile.mktemp(suffix='.png')
    pygame.image.save(surf, f_path)
    try:
        reader = png.Reader(filename=f_path)
        reader.read()
        palette = reader.palette()
        self.assertEqual(tuple(next(palette)), reddish_pixel)
        self.assertEqual(tuple(next(palette)), greenish_pixel)
        self.assertEqual(tuple(next(palette)), bluish_pixel)
        self.assertEqual(tuple(next(palette)), greyish_pixel)
    finally:
        if not reader.file.closed:
            reader.file.close()
        del reader
        os.remove(f_path)