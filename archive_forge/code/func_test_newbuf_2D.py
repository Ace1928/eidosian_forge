import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
def test_newbuf_2D(self):
    buftools = self.buftools
    Importer = buftools.Importer
    for bit_size in [8, 16, 24, 32]:
        s = pygame.Surface((10, 2), 0, bit_size)
        ar = pygame.PixelArray(s)
        format = self.bitsize_to_format[bit_size]
        itemsize = ar.itemsize
        shape = ar.shape
        w, h = shape
        strides = ar.strides
        length = w * h * itemsize
        imp = Importer(ar, buftools.PyBUF_FULL)
        self.assertTrue(imp.obj, ar)
        self.assertEqual(imp.len, length)
        self.assertEqual(imp.ndim, 2)
        self.assertEqual(imp.itemsize, itemsize)
        self.assertEqual(imp.format, format)
        self.assertFalse(imp.readonly)
        self.assertEqual(imp.shape, shape)
        self.assertEqual(imp.strides, strides)
        self.assertTrue(imp.suboffsets is None)
        self.assertEqual(imp.buf, s._pixels_address)
    s = pygame.Surface((8, 16), 0, 32)
    ar = pygame.PixelArray(s)
    format = self.bitsize_to_format[s.get_bitsize()]
    itemsize = ar.itemsize
    shape = ar.shape
    w, h = shape
    strides = ar.strides
    length = w * h * itemsize
    imp = Importer(ar, buftools.PyBUF_SIMPLE)
    self.assertTrue(imp.obj, ar)
    self.assertEqual(imp.len, length)
    self.assertEqual(imp.ndim, 0)
    self.assertEqual(imp.itemsize, itemsize)
    self.assertTrue(imp.format is None)
    self.assertFalse(imp.readonly)
    self.assertTrue(imp.shape is None)
    self.assertTrue(imp.strides is None)
    self.assertTrue(imp.suboffsets is None)
    self.assertEqual(imp.buf, s._pixels_address)
    imp = Importer(ar, buftools.PyBUF_FORMAT)
    self.assertEqual(imp.ndim, 0)
    self.assertEqual(imp.format, format)
    imp = Importer(ar, buftools.PyBUF_WRITABLE)
    self.assertEqual(imp.ndim, 0)
    self.assertTrue(imp.format is None)
    imp = Importer(ar, buftools.PyBUF_F_CONTIGUOUS)
    self.assertEqual(imp.ndim, 2)
    self.assertTrue(imp.format is None)
    self.assertEqual(imp.shape, shape)
    self.assertEqual(imp.strides, strides)
    imp = Importer(ar, buftools.PyBUF_ANY_CONTIGUOUS)
    self.assertEqual(imp.ndim, 2)
    self.assertTrue(imp.format is None)
    self.assertEqual(imp.shape, shape)
    self.assertEqual(imp.strides, strides)
    self.assertRaises(BufferError, Importer, ar, buftools.PyBUF_C_CONTIGUOUS)
    self.assertRaises(BufferError, Importer, ar, buftools.PyBUF_ND)
    ar_sliced = ar[:, ::2]
    format = self.bitsize_to_format[s.get_bitsize()]
    itemsize = ar_sliced.itemsize
    shape = ar_sliced.shape
    w, h = shape
    strides = ar_sliced.strides
    length = w * h * itemsize
    imp = Importer(ar_sliced, buftools.PyBUF_STRIDED)
    self.assertEqual(imp.len, length)
    self.assertEqual(imp.ndim, 2)
    self.assertEqual(imp.itemsize, itemsize)
    self.assertTrue(imp.format is None)
    self.assertFalse(imp.readonly)
    self.assertEqual(imp.shape, shape)
    self.assertEqual(imp.strides, strides)
    self.assertEqual(imp.buf, s._pixels_address)
    self.assertRaises(BufferError, Importer, ar_sliced, buftools.PyBUF_SIMPLE)
    self.assertRaises(BufferError, Importer, ar_sliced, buftools.PyBUF_ND)
    self.assertRaises(BufferError, Importer, ar_sliced, buftools.PyBUF_C_CONTIGUOUS)
    self.assertRaises(BufferError, Importer, ar_sliced, buftools.PyBUF_F_CONTIGUOUS)
    self.assertRaises(BufferError, Importer, ar_sliced, buftools.PyBUF_ANY_CONTIGUOUS)
    ar_sliced = ar[::2, :]
    format = self.bitsize_to_format[s.get_bitsize()]
    itemsize = ar_sliced.itemsize
    shape = ar_sliced.shape
    w, h = shape
    strides = ar_sliced.strides
    length = w * h * itemsize
    imp = Importer(ar_sliced, buftools.PyBUF_STRIDED)
    self.assertEqual(imp.len, length)
    self.assertEqual(imp.ndim, 2)
    self.assertEqual(imp.itemsize, itemsize)
    self.assertTrue(imp.format is None)
    self.assertFalse(imp.readonly)
    self.assertEqual(imp.shape, shape)
    self.assertEqual(imp.strides, strides)
    self.assertEqual(imp.buf, s._pixels_address)
    self.assertRaises(BufferError, Importer, ar_sliced, buftools.PyBUF_SIMPLE)
    self.assertRaises(BufferError, Importer, ar_sliced, buftools.PyBUF_ND)
    self.assertRaises(BufferError, Importer, ar_sliced, buftools.PyBUF_C_CONTIGUOUS)
    self.assertRaises(BufferError, Importer, ar_sliced, buftools.PyBUF_F_CONTIGUOUS)
    self.assertRaises(BufferError, Importer, ar_sliced, buftools.PyBUF_ANY_CONTIGUOUS)
    s2 = s.subsurface((2, 3, 5, 7))
    ar = pygame.PixelArray(s2)
    format = self.bitsize_to_format[s.get_bitsize()]
    itemsize = ar.itemsize
    shape = ar.shape
    w, h = shape
    strides = ar.strides
    length = w * h * itemsize
    imp = Importer(ar, buftools.PyBUF_STRIDES)
    self.assertTrue(imp.obj, ar)
    self.assertEqual(imp.len, length)
    self.assertEqual(imp.ndim, 2)
    self.assertEqual(imp.itemsize, itemsize)
    self.assertTrue(imp.format is None)
    self.assertFalse(imp.readonly)
    self.assertEqual(imp.shape, shape)
    self.assertEqual(imp.strides, strides)
    self.assertTrue(imp.suboffsets is None)
    self.assertEqual(imp.buf, s2._pixels_address)
    self.assertRaises(BufferError, Importer, ar, buftools.PyBUF_SIMPLE)
    self.assertRaises(BufferError, Importer, ar, buftools.PyBUF_FORMAT)
    self.assertRaises(BufferError, Importer, ar, buftools.PyBUF_WRITABLE)
    self.assertRaises(BufferError, Importer, ar, buftools.PyBUF_ND)
    self.assertRaises(BufferError, Importer, ar, buftools.PyBUF_C_CONTIGUOUS)
    self.assertRaises(BufferError, Importer, ar, buftools.PyBUF_F_CONTIGUOUS)
    self.assertRaises(BufferError, Importer, ar, buftools.PyBUF_ANY_CONTIGUOUS)