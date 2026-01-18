import unittest
import pygame.constants
def test_kmod__type(self):
    """Ensures KMOD constants are the correct type."""
    for name in self.KMOD_CONSTANTS:
        value = getattr(pygame.constants, name)
        self.assertIs(type(value), int)