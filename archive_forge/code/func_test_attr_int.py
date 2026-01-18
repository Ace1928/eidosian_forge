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
def test_attr_int(self) -> None:
    attr = helper.make_attribute('int', 3)
    self.assertEqual(attr.name, 'int')
    self.assertEqual(attr.i, 3)
    checker.check_attribute(attr)
    attr = helper.make_attribute('int', 5)
    self.assertEqual(attr.name, 'int')
    self.assertEqual(attr.i, 5)
    checker.check_attribute(attr)
    attr = helper.make_attribute('int', 961)
    self.assertEqual(attr.name, 'int')
    self.assertEqual(attr.i, 961)
    checker.check_attribute(attr)
    attr = helper.make_attribute('int', 5889)
    self.assertEqual(attr.name, 'int')
    self.assertEqual(attr.i, 5889)
    checker.check_attribute(attr)