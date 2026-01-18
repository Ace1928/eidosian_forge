import collections
import time
import unittest
import os
import pygame
def test_custom_type__reset(self):
    """Ensure custom events get 'deregistered' by quit()."""
    before = pygame.event.custom_type()
    self.assertEqual(before, pygame.event.custom_type() - 1)
    pygame.quit()
    pygame.init()
    pygame.display.init()
    self.assertEqual(before, pygame.event.custom_type())