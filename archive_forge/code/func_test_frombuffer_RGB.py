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
def test_frombuffer_RGB(self):
    rgb_buffer = bytearray([255, 10, 20, 255, 10, 20, 255, 10, 20, 255, 10, 20, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 200, 20, 50, 200, 20, 50, 200, 20, 50, 200, 20])
    rgb_surf = pygame.image.frombuffer(rgb_buffer, (4, 4), 'RGB')
    self.assertEqual(rgb_surf.get_at((0, 0)), pygame.Color(255, 10, 20))
    self.assertEqual(rgb_surf.get_at((1, 1)), pygame.Color(255, 255, 255))
    self.assertEqual(rgb_surf.get_at((2, 2)), pygame.Color(0, 0, 0))
    self.assertEqual(rgb_surf.get_at((3, 3)), pygame.Color(50, 200, 20))