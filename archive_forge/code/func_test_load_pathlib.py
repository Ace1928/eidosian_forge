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
def test_load_pathlib(self):
    """works loading using a Path argument."""
    path = pathlib.Path(example_path('data/asprite.bmp'))
    surf = pygame.image.load_extended(path)
    self.assertEqual(surf.get_at((0, 0)), (255, 255, 255, 255))