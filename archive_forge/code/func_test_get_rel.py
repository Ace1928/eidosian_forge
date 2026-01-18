import unittest
import os
import platform
import warnings
import pygame
def test_get_rel(self):
    """Ensures get_rel returns the correct types."""
    expected_length = 2
    rel = pygame.mouse.get_rel()
    self.assertIsInstance(rel, tuple)
    self.assertEqual(len(rel), expected_length)
    for value in rel:
        self.assertIsInstance(value, int)