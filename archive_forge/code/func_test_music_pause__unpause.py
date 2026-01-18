import os
import sys
import platform
import unittest
import time
from pygame.tests.test_utils import example_path
import pygame
def test_music_pause__unpause(self):
    """Ensure music has the correct position immediately after unpausing

        |tags:music|
        """
    filename = example_path(os.path.join('data', 'house_lo.mp3'))
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    time.sleep(0.05)
    pygame.mixer.music.pause()
    time.sleep(0.05)
    before_unpause = pygame.mixer.music.get_pos()
    pygame.mixer.music.unpause()
    after_unpause = pygame.mixer.music.get_pos()
    self.assertEqual(before_unpause, after_unpause)