import sys
import unittest
import platform
import pygame
def init_assertions(self):
    self.assertTrue(pygame.get_init())
    self.assertTrue(pygame.display.get_init())
    if 'pygame.mixer' in sys.modules:
        self.assertTrue(pygame.mixer.get_init())
    if 'pygame.font' in sys.modules:
        self.assertTrue(pygame.font.get_init())