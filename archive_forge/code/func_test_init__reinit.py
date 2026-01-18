import os
import sys
import unittest
from pygame.tests.test_utils import trunk_relative_path
import pygame
from pygame import scrap
def test_init__reinit(self):
    """Ensures reinitializing the scrap module doesn't clear its data."""
    data_type = pygame.SCRAP_TEXT
    expected_data = b'test_init__reinit'
    scrap.put(data_type, expected_data)
    scrap.init()
    self.assertEqual(scrap.get(data_type), expected_data)