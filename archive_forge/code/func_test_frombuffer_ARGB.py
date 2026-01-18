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
def test_frombuffer_ARGB(self):
    argb_buffer = bytearray([200, 255, 10, 20, 200, 255, 10, 20, 200, 255, 10, 20, 200, 255, 10, 20, 127, 255, 255, 255, 127, 255, 255, 255, 127, 255, 255, 255, 127, 255, 255, 255, 79, 0, 0, 0, 79, 0, 0, 0, 79, 0, 0, 0, 79, 0, 0, 0, 255, 50, 200, 20, 255, 50, 200, 20, 255, 50, 200, 20, 255, 50, 200, 20])
    argb_surf = pygame.image.frombuffer(argb_buffer, (4, 4), 'ARGB')
    self.assertEqual(argb_surf.get_at((0, 0)), pygame.Color(255, 10, 20, 200))
    self.assertEqual(argb_surf.get_at((1, 1)), pygame.Color(255, 255, 255, 127))
    self.assertEqual(argb_surf.get_at((2, 2)), pygame.Color(0, 0, 0, 79))
    self.assertEqual(argb_surf.get_at((3, 3)), pygame.Color(50, 200, 20, 255))