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
def test_frombuffer_BGR(self):
    bgr_buffer = bytearray([20, 10, 255, 20, 10, 255, 20, 10, 255, 20, 10, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 200, 50, 20, 200, 50, 20, 200, 50, 20, 200, 50])
    bgr_surf = pygame.image.frombuffer(bgr_buffer, (4, 4), 'BGR')
    self.assertEqual(bgr_surf.get_at((0, 0)), pygame.Color(255, 10, 20))
    self.assertEqual(bgr_surf.get_at((1, 1)), pygame.Color(255, 255, 255))
    self.assertEqual(bgr_surf.get_at((2, 2)), pygame.Color(0, 0, 0))
    self.assertEqual(bgr_surf.get_at((3, 3)), pygame.Color(50, 200, 20))