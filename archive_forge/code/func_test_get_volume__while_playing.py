import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_get_volume__while_playing(self):
    """Ensure a sound's volume can be retrieved while playing."""
    try:
        expected_volume = 1.0
        filename = example_path(os.path.join('data', 'house_lo.wav'))
        sound = mixer.Sound(file=filename)
        sound.play(-1)
        volume = sound.get_volume()
        self.assertAlmostEqual(volume, expected_volume)
    finally:
        pygame.mixer.quit()
        with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
            sound.get_volume()