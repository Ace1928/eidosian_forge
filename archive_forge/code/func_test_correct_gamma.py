import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_correct_gamma(self):
    mc1 = self.MyColor(64, 70, 75, 255)
    self.assertTrue(mc1.an_attribute)
    mc2 = mc1.correct_gamma(0.03)
    self.assertTrue(isinstance(mc2, self.MyColor))
    self.assertRaises(AttributeError, getattr, mc2, 'an_attribute')