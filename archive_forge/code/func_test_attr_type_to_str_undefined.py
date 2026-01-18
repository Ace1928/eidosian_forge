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
def test_attr_type_to_str_undefined(self):
    result = helper._attr_type_to_str(9999)
    self.assertEqual(result, 'UNDEFINED')