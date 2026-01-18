import os
import unittest
from pygame.tests import test_utils
from pygame.tests.test_utils import (
import pygame
from pygame.locals import *
from pygame.bufferproxy import BufferProxy
import platform
import gc
import weakref
import ctypes
def test_fill_keyword_args(self):
    """Ensure fill() accepts keyword arguments."""
    color = (1, 2, 3, 255)
    area = (1, 1, 2, 2)
    s1 = pygame.Surface((4, 4), 0, 32)
    s1.fill(special_flags=pygame.BLEND_ADD, color=color, rect=area)
    self.assertEqual(s1.get_at((0, 0)), (0, 0, 0, 255))
    self.assertEqual(s1.get_at((1, 1)), color)