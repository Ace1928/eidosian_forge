import unittest
from traits.api import (
from traits.testing.optional_dependencies import numpy, requires_numpy
@requires_numpy
def test_accepts_numpy_types(self):
    numpy_values = [numpy.uint8(25), numpy.uint16(25), numpy.uint32(25), numpy.uint64(25), numpy.int8(25), numpy.int16(25), numpy.int32(25), numpy.int64(25), numpy.float16(25), numpy.float32(25), numpy.float64(25)]
    for numpy_value in numpy_values:
        self.model.percentage = numpy_value
        self.assertIs(type(self.model.percentage), float)
        self.assertEqual(self.model.percentage, 25.0)