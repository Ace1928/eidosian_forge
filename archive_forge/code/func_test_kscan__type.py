import unittest
import pygame.constants
def test_kscan__type(self):
    """Ensures KSCAN constants are the correct type."""
    for name in self.KSCAN_NAMES:
        value = getattr(pygame.constants, name)
        self.assertIs(type(value), int)