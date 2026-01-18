import re
import weakref
import gc
import ctypes
import unittest
import pygame
from pygame.bufferproxy import BufferProxy
def test___array_struct___property(self):
    kwds = self.view_keywords
    v = BufferProxy(kwds)
    d = pygame.get_array_interface(v)
    self.assertEqual(len(d), 5)
    self.assertEqual(d['version'], 3)
    self.assertEqual(d['shape'], kwds['shape'])
    self.assertEqual(d['typestr'], kwds['typestr'])
    self.assertEqual(d['data'], kwds['data'])
    self.assertEqual(d['strides'], kwds['strides'])