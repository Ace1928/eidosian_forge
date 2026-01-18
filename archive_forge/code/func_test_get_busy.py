import os
import sys
import platform
import unittest
import time
from pygame.tests.test_utils import example_path
import pygame
def test_get_busy(self):
    self.music_load('ogg')
    self.assertFalse(pygame.mixer.music.get_busy())
    pygame.mixer.music.play()
    self.assertTrue(pygame.mixer.music.get_busy())
    pygame.mixer.music.pause()
    self.assertFalse(pygame.mixer.music.get_busy())