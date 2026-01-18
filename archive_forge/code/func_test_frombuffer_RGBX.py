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
def test_frombuffer_RGBX(self):
    rgbx_buffer = bytearray([255, 10, 20, 255, 255, 10, 20, 255, 255, 10, 20, 255, 255, 10, 20, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 50, 200, 20, 255, 50, 200, 20, 255, 50, 200, 20, 255, 50, 200, 20, 255])
    rgbx_surf = pygame.image.frombuffer(rgbx_buffer, (4, 4), 'RGBX')
    self.assertEqual(rgbx_surf.get_at((0, 0)), pygame.Color(255, 10, 20, 255))
    self.assertEqual(rgbx_surf.get_at((1, 1)), pygame.Color(255, 255, 255, 255))
    self.assertEqual(rgbx_surf.get_at((2, 2)), pygame.Color(0, 0, 0, 255))
    self.assertEqual(rgbx_surf.get_at((3, 3)), pygame.Color(50, 200, 20, 255))