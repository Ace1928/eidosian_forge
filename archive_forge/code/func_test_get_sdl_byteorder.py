import sys
import unittest
import platform
import pygame
def test_get_sdl_byteorder(self):
    """Ensure the SDL byte order is valid"""
    byte_order = pygame.get_sdl_byteorder()
    expected_options = (pygame.LIL_ENDIAN, pygame.BIG_ENDIAN)
    self.assertIn(byte_order, expected_options)