import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_issue_269(self):
    """PyColor OverflowError on HSVA with hue value of 360

        >>> c = pygame.Color(0)
        >>> c.hsva = (360,0,0,0)
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        OverflowError: this is not allowed to happen ever
        >>> pygame.ver
        '1.9.1release'
        >>>

        """
    c = pygame.Color(0)
    c.hsva = (360, 0, 0, 0)
    self.assertEqual(c.hsva, (0, 0, 0, 0))
    c.hsva = (360, 100, 100, 100)
    self.assertEqual(c.hsva, (0, 100, 100, 100))
    self.assertEqual(c, (255, 0, 0, 255))