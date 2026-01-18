import re
import weakref
import gc
import ctypes
import unittest
import pygame
from pygame.bufferproxy import BufferProxy
def test___array_interface___property(self):
    kwds = self.view_keywords
    v = BufferProxy(kwds)
    d = v.__array_interface__
    self.assertEqual(len(d), 5)
    self.assertEqual(d['version'], 3)
    self.assertEqual(d['shape'], kwds['shape'])
    self.assertEqual(d['typestr'], kwds['typestr'])
    self.assertEqual(d['data'], kwds['data'])
    self.assertEqual(d['strides'], kwds['strides'])