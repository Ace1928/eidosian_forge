from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_from_threshold(self):
    """Does mask.from_threshold() work correctly?"""
    a = [16, 24, 32]
    for i in a:
        surf = pygame.surface.Surface((70, 70), 0, i)
        surf.fill((100, 50, 200), (20, 20, 20, 20))
        mask = pygame.mask.from_threshold(surf, (100, 50, 200, 255), (10, 10, 10, 255))
        rects = mask.get_bounding_rects()
        self.assertEqual(mask.count(), 400)
        self.assertEqual(mask.get_bounding_rects(), [pygame.Rect((20, 20, 20, 20))])
    for i in a:
        surf = pygame.surface.Surface((70, 70), 0, i)
        surf2 = pygame.surface.Surface((70, 70), 0, i)
        surf.fill((100, 100, 100))
        surf2.fill((150, 150, 150))
        surf2.fill((100, 100, 100), (40, 40, 10, 10))
        mask = pygame.mask.from_threshold(surface=surf, color=(0, 0, 0, 0), threshold=(10, 10, 10, 255), othersurface=surf2)
        self.assertIsInstance(mask, pygame.mask.Mask)
        self.assertEqual(mask.count(), 100)
        self.assertEqual(mask.get_bounding_rects(), [pygame.Rect((40, 40, 10, 10))])