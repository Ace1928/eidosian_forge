import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_average_surfaces__24(self):
    SIZE = 32
    depth = 24
    s1 = pygame.Surface((SIZE, SIZE), 0, depth)
    s2 = pygame.Surface((SIZE, SIZE), 0, depth)
    s3 = pygame.Surface((SIZE, SIZE), 0, depth)
    s1.fill((10, 10, 70, 255))
    s2.fill((10, 20, 70, 255))
    s3.fill((10, 130, 10, 255))
    surfaces = [s1, s2, s3]
    sr = pygame.transform.average_surfaces(surfaces)
    self.assertEqual(sr.get_masks(), s1.get_masks())
    self.assertEqual(sr.get_flags(), s1.get_flags())
    self.assertEqual(sr.get_losses(), s1.get_losses())
    if 0:
        print(sr, s1)
        print(sr.get_masks(), s1.get_masks())
        print(sr.get_flags(), s1.get_flags())
        print(sr.get_losses(), s1.get_losses())
        print(sr.get_shifts(), s1.get_shifts())
    self.assertEqual(sr.get_at((0, 0)), (10, 53, 50, 255))