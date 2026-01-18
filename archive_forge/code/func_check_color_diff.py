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
def check_color_diff(color1, color2):
    """Returns True if two colors are within (1, 1, 1, 1) of each other."""
    for val in color1 - color2:
        if abs(val) > 1:
            return False
    return True