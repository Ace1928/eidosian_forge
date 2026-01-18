import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_grayscale(self):
    Color = pygame.color.Color
    color = Color(255, 0, 0, 255)
    self.assertEqual(color.grayscale(), Color(76, 76, 76, 255))
    color = Color(3, 5, 7, 255)
    self.assertEqual(color.grayscale(), Color(4, 4, 4, 255))
    color = Color(3, 5, 70, 255)
    self.assertEqual(color.grayscale(), Color(11, 11, 11, 255))
    color = Color(3, 50, 70, 255)
    self.assertEqual(color.grayscale(), Color(38, 38, 38, 255))
    color = Color(30, 50, 70, 255)
    self.assertEqual(color.grayscale(), Color(46, 46, 46, 255))
    color = Color(255, 0, 0, 144)
    self.assertEqual(color.grayscale(), Color(76, 76, 76, 144))
    color = Color(3, 5, 7, 144)
    self.assertEqual(color.grayscale(), Color(4, 4, 4, 144))
    color = Color(3, 5, 70, 144)
    self.assertEqual(color.grayscale(), Color(11, 11, 11, 144))
    color = Color(3, 50, 70, 144)
    self.assertEqual(color.grayscale(), Color(38, 38, 38, 144))
    color = Color(30, 50, 70, 144)
    self.assertEqual(color.grayscale(), Color(46, 46, 46, 144))