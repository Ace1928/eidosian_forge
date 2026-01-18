import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_decodeFloatingPointAdditionalTests(self):
    self.assertAlmostEqual(-1.1234567893, ujson.loads('-1.1234567893'))
    self.assertAlmostEqual(-1.234567893, ujson.loads('-1.234567893'))
    self.assertAlmostEqual(-1.34567893, ujson.loads('-1.34567893'))
    self.assertAlmostEqual(-1.4567893, ujson.loads('-1.4567893'))
    self.assertAlmostEqual(-1.567893, ujson.loads('-1.567893'))
    self.assertAlmostEqual(-1.67893, ujson.loads('-1.67893'))
    self.assertAlmostEqual(-1.7894, ujson.loads('-1.7894'))
    self.assertAlmostEqual(-1.893, ujson.loads('-1.893'))
    self.assertAlmostEqual(-1.3, ujson.loads('-1.3'))
    self.assertAlmostEqual(1.1234567893, ujson.loads('1.1234567893'))
    self.assertAlmostEqual(1.234567893, ujson.loads('1.234567893'))
    self.assertAlmostEqual(1.34567893, ujson.loads('1.34567893'))
    self.assertAlmostEqual(1.4567893, ujson.loads('1.4567893'))
    self.assertAlmostEqual(1.567893, ujson.loads('1.567893'))
    self.assertAlmostEqual(1.67893, ujson.loads('1.67893'))
    self.assertAlmostEqual(1.7894, ujson.loads('1.7894'))
    self.assertAlmostEqual(1.893, ujson.loads('1.893'))
    self.assertAlmostEqual(1.3, ujson.loads('1.3'))