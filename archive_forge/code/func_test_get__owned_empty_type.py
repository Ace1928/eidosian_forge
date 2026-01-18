import os
import sys
import unittest
from pygame.tests.test_utils import trunk_relative_path
import pygame
from pygame import scrap
def test_get__owned_empty_type(self):
    """Ensures get works when there is no data of the requested type
        in the clipboard and the clipboard is owned by the pygame application.
        """
    DATA_TYPE = 'test_get__owned_empty_type'
    if scrap.lost():
        scrap.put(pygame.SCRAP_TEXT, b'text to clipboard')
        if scrap.lost():
            self.skipTest('requires the pygame application to own the clipboard')
    data = scrap.get(DATA_TYPE)
    self.assertIsNone(data)