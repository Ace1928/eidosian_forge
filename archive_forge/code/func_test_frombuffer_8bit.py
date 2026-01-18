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
def test_frombuffer_8bit(self):
    """test reading pixel data from a bytes buffer"""
    pygame.display.init()
    eight_bit_palette_buffer = bytearray([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    eight_bit_surf = pygame.image.frombuffer(eight_bit_palette_buffer, (4, 4), 'P')
    eight_bit_surf.set_palette([(255, 10, 20), (255, 255, 255), (0, 0, 0), (50, 200, 20)])
    self.assertEqual(eight_bit_surf.get_at((0, 0)), pygame.Color(255, 10, 20))
    self.assertEqual(eight_bit_surf.get_at((1, 1)), pygame.Color(255, 255, 255))
    self.assertEqual(eight_bit_surf.get_at((2, 2)), pygame.Color(0, 0, 0))
    self.assertEqual(eight_bit_surf.get_at((3, 3)), pygame.Color(50, 200, 20))