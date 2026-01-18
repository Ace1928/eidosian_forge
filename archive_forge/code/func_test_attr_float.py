import itertools
import random
import struct
import unittest
from typing import Any, List, Tuple
import numpy as np
import parameterized
import pytest
import version_utils
from onnx import (
from onnx.reference.op_run import to_array_extended
def test_attr_float(self) -> None:
    attr = helper.make_attribute('float', 1.0)
    self.assertEqual(attr.name, 'float')
    self.assertEqual(attr.f, 1.0)
    checker.check_attribute(attr)
    attr = helper.make_attribute('float', 10000000000.0)
    self.assertEqual(attr.name, 'float')
    self.assertEqual(attr.f, 10000000000.0)
    checker.check_attribute(attr)