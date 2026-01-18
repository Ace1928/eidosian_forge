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
def test_flags_default0_nodisplay(self):
    """is set to zero, and SRCALPHA is not set by default with no display initialized."""
    pygame.display.quit()
    surf = pygame.Surface((70, 70))
    self.assertEqual(surf.get_flags() & SRCALPHA, 0)