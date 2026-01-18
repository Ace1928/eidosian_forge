import unittest
import pygame
from pygame.locals import *
def test_blits_wrong_length(self):
    dst = pygame.Surface((100, 10), SRCALPHA, 32)
    self.assertRaises(ValueError, dst.blits, [pygame.Surface((10, 10), SRCALPHA, 32)])