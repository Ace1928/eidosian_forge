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
def blit_locked_test(surface):
    newSurf = pygame.Surface((10, 10))
    try:
        newSurf.blit(surface, (0, 0))
    except pygame.error:
        return True
    else:
        return False