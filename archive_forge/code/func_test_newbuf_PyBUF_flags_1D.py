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
@unittest.skipIf(not pygame.HAVE_NEWBUF, 'newbuf not implemented')
def test_newbuf_PyBUF_flags_1D(self):
    from pygame.tests.test_utils import buftools
    Importer = buftools.Importer
    s = pygame.Surface((10, 6), 0, 32)
    a = s.get_view('1')
    b = Importer(a, buftools.PyBUF_SIMPLE)
    self.assertEqual(b.ndim, 0)
    self.assertTrue(b.format is None)
    self.assertEqual(b.len, a.length)
    self.assertEqual(b.itemsize, s.get_bytesize())
    self.assertTrue(b.shape is None)
    self.assertTrue(b.strides is None)
    self.assertTrue(b.suboffsets is None)
    self.assertFalse(b.readonly)
    self.assertEqual(b.buf, s._pixels_address)
    b = Importer(a, buftools.PyBUF_WRITABLE)
    self.assertEqual(b.ndim, 0)
    self.assertTrue(b.format is None)
    self.assertFalse(b.readonly)
    b = Importer(a, buftools.PyBUF_FORMAT)
    self.assertEqual(b.ndim, 0)
    self.assertEqual(b.format, '=I')
    b = Importer(a, buftools.PyBUF_ND)
    self.assertEqual(b.ndim, 1)
    self.assertTrue(b.format is None)
    self.assertEqual(b.len, a.length)
    self.assertEqual(b.itemsize, s.get_bytesize())
    self.assertEqual(b.shape, (s.get_width() * s.get_height(),))
    self.assertTrue(b.strides is None)
    self.assertTrue(b.suboffsets is None)
    self.assertFalse(b.readonly)
    self.assertEqual(b.buf, s._pixels_address)
    b = Importer(a, buftools.PyBUF_STRIDES)
    self.assertEqual(b.ndim, 1)
    self.assertTrue(b.format is None)
    self.assertEqual(b.strides, (s.get_bytesize(),))