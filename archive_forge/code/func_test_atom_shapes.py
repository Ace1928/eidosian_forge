from __future__ import absolute_import, division, print_function
import ctypes
import unittest
import pytest
from datashader import datashape
from datashader.datashape import dshape, error, DataShape, Record
def test_atom_shapes(self):
    self.assertEqual(dshape('bool'), dshape(datashape.bool_))
    self.assertEqual(dshape('int8'), dshape(datashape.int8))
    self.assertEqual(dshape('int16'), dshape(datashape.int16))
    self.assertEqual(dshape('int32'), dshape(datashape.int32))
    self.assertEqual(dshape('int64'), dshape(datashape.int64))
    self.assertEqual(dshape('uint8'), dshape(datashape.uint8))
    self.assertEqual(dshape('uint16'), dshape(datashape.uint16))
    self.assertEqual(dshape('uint32'), dshape(datashape.uint32))
    self.assertEqual(dshape('uint64'), dshape(datashape.uint64))
    self.assertEqual(dshape('float32'), dshape(datashape.float32))
    self.assertEqual(dshape('float64'), dshape(datashape.float64))
    self.assertEqual(dshape('complex64'), dshape(datashape.complex64))
    self.assertEqual(dshape('complex128'), dshape(datashape.complex128))
    self.assertEqual(dshape('complex64'), dshape('complex[float32]'))
    self.assertEqual(dshape('complex128'), dshape('complex[float64]'))
    self.assertEqual(dshape('string'), dshape(datashape.string))
    self.assertEqual(dshape('json'), dshape(datashape.json))
    self.assertEqual(dshape('date'), dshape(datashape.date_))
    self.assertEqual(dshape('time'), dshape(datashape.time_))
    self.assertEqual(dshape('datetime'), dshape(datashape.datetime_))