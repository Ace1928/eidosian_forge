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
@unittest.skipIf(not OLDBUF, 'old buffer not available')
def test_get_buffer_oldbuf(self):
    from pygame.bufferproxy import get_segcount, get_write_buffer
    s = pygame.Surface((2, 4), pygame.SRCALPHA, 32)
    v = s.get_buffer()
    segcount, buflen = get_segcount(v)
    self.assertEqual(segcount, 1)
    self.assertEqual(buflen, s.get_pitch() * s.get_height())
    seglen, segaddr = get_write_buffer(v, 0)
    self.assertEqual(segaddr, s._pixels_address)
    self.assertEqual(seglen, buflen)