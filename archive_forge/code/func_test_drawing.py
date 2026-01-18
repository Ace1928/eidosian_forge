from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_drawing(self):
    """Test fill, clear, invert, draw, erase"""
    m = pygame.Mask((100, 100))
    self.assertEqual(m.count(), 0)
    m.fill()
    self.assertEqual(m.count(), 10000)
    m2 = pygame.Mask((10, 10), fill=True)
    m.erase(m2, (50, 50))
    self.assertEqual(m.count(), 9900)
    m.invert()
    self.assertEqual(m.count(), 100)
    m.draw(m2, (0, 0))
    self.assertEqual(m.count(), 200)
    m.clear()
    self.assertEqual(m.count(), 0)