import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_get_sdl_mixer_version__kwargs(self):
    """Ensures get_sdl_mixer_version works correctly using kwargs."""
    expected_length = 3
    expected_type = tuple
    expected_item_type = int
    for value in (True, False):
        version = pygame.mixer.get_sdl_mixer_version(linked=value)
        self.assertIsInstance(version, expected_type)
        self.assertEqual(len(version), expected_length)
        for item in version:
            self.assertIsInstance(item, expected_item_type)