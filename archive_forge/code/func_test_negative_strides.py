import a new buffer interface.
import pygame
import pygame.newbuffer
from pygame.newbuffer import (
import unittest
import ctypes
import operator
from functools import reduce
def test_negative_strides(self):
    self.check_args(3, (3, 5, 4), 'B', (20, 4, -1), 60, 60, 1, 3)
    self.check_args(3, (3, 5, 3), 'B', (20, 4, -1), 45, 60, 1, 2)
    self.check_args(3, (3, 5, 4), 'B', (20, -4, 1), 60, 60, 1, 16)
    self.check_args(3, (3, 5, 4), 'B', (-20, -4, -1), 60, 60, 1, 59)
    self.check_args(3, (3, 5, 3), 'B', (-20, -4, -1), 45, 60, 1, 58)