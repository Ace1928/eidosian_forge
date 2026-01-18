import unittest
import pygame.constants
def test_k__existence(self):
    """Ensures K constants exist."""
    for name in self.K_NAMES:
        self.assertTrue(hasattr(pygame.constants, name), f'missing constant {name}')