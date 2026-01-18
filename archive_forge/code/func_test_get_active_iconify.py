import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
@unittest.skipIf(os.environ.get('SDL_VIDEODRIVER') == 'dummy', 'requires the SDL_VIDEODRIVER to be a non dummy value')
def test_get_active_iconify(self):
    """Test the get_active function after an iconify"""
    pygame.display.set_mode((640, 480))
    pygame.event.clear()
    pygame.display.iconify()
    for _ in range(100):
        time.sleep(0.01)
        pygame.event.pump()
    self.assertEqual(pygame.display.get_active(), False)