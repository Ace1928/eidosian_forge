import a new buffer interface.
import pygame
import pygame.newbuffer
from pygame.newbuffer import (
import unittest
import ctypes
import operator
from functools import reduce
@property
def suboffsets(self):
    """return int tuple or None for NULL field"""
    return self._to_ssize_tuple(self._view.suboffsets)