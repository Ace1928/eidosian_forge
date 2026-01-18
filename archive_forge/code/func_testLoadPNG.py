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
def testLoadPNG(self):
    """see if we can load a png with color values in the proper channels."""
    reddish_pixel = (210, 0, 0, 255)
    greenish_pixel = (0, 220, 0, 255)
    bluish_pixel = (0, 0, 230, 255)
    greyish_pixel = (110, 120, 130, 140)
    pixel_array = [reddish_pixel + greenish_pixel, bluish_pixel + greyish_pixel]
    f_descriptor, f_path = tempfile.mkstemp(suffix='.png')
    with os.fdopen(f_descriptor, 'wb') as f:
        w = png.Writer(2, 2, alpha=True)
        w.write(f, pixel_array)
    surf = pygame.image.load(f_path)
    self.assertEqual(surf.get_at((0, 0)), reddish_pixel)
    self.assertEqual(surf.get_at((1, 0)), greenish_pixel)
    self.assertEqual(surf.get_at((0, 1)), bluish_pixel)
    self.assertEqual(surf.get_at((1, 1)), greyish_pixel)
    with open(f_path, 'rb') as f:
        surf = pygame.image.load(f)
    self.assertEqual(surf.get_at((0, 0)), reddish_pixel)
    self.assertEqual(surf.get_at((1, 0)), greenish_pixel)
    self.assertEqual(surf.get_at((0, 1)), bluish_pixel)
    self.assertEqual(surf.get_at((1, 1)), greyish_pixel)
    os.remove(f_path)