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
def test_blit__SRCALPHA32_to_8(self):
    target = pygame.Surface((11, 8), 0, 8)
    test_color = target.get_palette_at(2)
    source = pygame.Surface((1, 1), pygame.SRCALPHA, 32)
    source.set_at((0, 0), test_color)
    target.blit(source, (0, 0))