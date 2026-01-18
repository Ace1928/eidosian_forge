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
def test_blit_keyword_args(self):
    color = (1, 2, 3, 255)
    s1 = pygame.Surface((4, 4), 0, 32)
    s2 = pygame.Surface((2, 2), 0, 32)
    s2.fill((1, 2, 3))
    s1.blit(special_flags=BLEND_ADD, source=s2, dest=(1, 1), area=s2.get_rect())
    self.assertEqual(s1.get_at((0, 0)), (0, 0, 0, 255))
    self.assertEqual(s1.get_at((1, 1)), color)