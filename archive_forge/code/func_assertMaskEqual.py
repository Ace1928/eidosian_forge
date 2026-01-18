from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def assertMaskEqual(testcase, m1, m2, msg=None):
    """Checks to see if the 2 given masks are equal."""
    m1_count = m1.count()
    testcase.assertEqual(m1.get_size(), m2.get_size(), msg=msg)
    testcase.assertEqual(m1_count, m2.count(), msg=msg)
    testcase.assertEqual(m1_count, m1.overlap_area(m2, (0, 0)), msg=msg)