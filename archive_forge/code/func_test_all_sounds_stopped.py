import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_all_sounds_stopped(self):
    """
        Test that get_busy returns False when all sounds are stopped.
        """
    sound1 = pygame.mixer.Sound(example_path('data/house_lo.wav'))
    sound2 = pygame.mixer.Sound(example_path('data/house_lo.wav'))
    sound1.play()
    sound2.play()
    time.sleep(0.2)
    sound1.stop()
    sound2.stop()
    time.sleep(0.2)
    self.assertFalse(pygame.mixer.get_busy())