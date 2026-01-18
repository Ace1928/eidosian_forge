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
def test_attr_mismatch(self) -> None:
    with self.assertRaisesRegex(TypeError, "Inferred attribute type 'FLOAT'"):
        helper.make_attribute('test', 6.4, attr_type=AttributeProto.STRING)