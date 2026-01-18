import unittest
import os
import platform
import warnings
import pygame
def test_get_pos(self):
    """Ensures get_pos returns the correct types."""
    expected_length = 2
    pos = pygame.mouse.get_pos()
    self.assertIsInstance(pos, tuple)
    self.assertEqual(len(pos), expected_length)
    for value in pos:
        self.assertIsInstance(value, int)