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
def test_node_with_arg(self) -> None:
    self.assertTrue(defs.has('Relu'))
    node_def = helper.make_node('Relu', ['X'], ['Y'], arg_value=1)
    self.assertEqual(node_def.op_type, 'Relu')
    self.assertEqual(list(node_def.input), ['X'])
    self.assertEqual(list(node_def.output), ['Y'])
    self.assertEqual(len(node_def.attribute), 1)
    self.assertEqual(node_def.attribute[0], helper.make_attribute('arg_value', 1))