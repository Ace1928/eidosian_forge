import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_get_sdl_mixer_version(self):
    """Ensures get_sdl_mixer_version works correctly with no args."""
    expected_length = 3
    expected_type = tuple
    expected_item_type = int
    version = pygame.mixer.get_sdl_mixer_version()
    self.assertIsInstance(version, expected_type)
    self.assertEqual(len(version), expected_length)
    for item in version:
        self.assertIsInstance(item, expected_item_type)