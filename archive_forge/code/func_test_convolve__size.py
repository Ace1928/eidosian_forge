from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_convolve__size(self):
    sizes = [(1, 1), (31, 31), (32, 32), (100, 100)]
    for s1 in sizes:
        m1 = pygame.Mask(s1)
        for s2 in sizes:
            m2 = pygame.Mask(s2)
            o = m1.convolve(m2)
            self.assertIsInstance(o, pygame.mask.Mask)
            for i in (0, 1):
                self.assertEqual(o.get_size()[i], m1.get_size()[i] + m2.get_size()[i] - 1)