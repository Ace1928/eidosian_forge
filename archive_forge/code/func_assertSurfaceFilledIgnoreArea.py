from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def assertSurfaceFilledIgnoreArea(testcase, surface, expected_color, ignore_rect):
    """Checks if the surface is filled with the given color. The
    ignore_rect area is not checked.
    """
    x_range = range(surface.get_width())
    y_range = range(surface.get_height())
    ignore_rect.normalize()
    surface.lock()
    for pos in ((x, y) for y in y_range for x in x_range):
        if not ignore_rect.collidepoint(pos):
            testcase.assertEqual(surface.get_at(pos), expected_color, pos)
    surface.unlock()