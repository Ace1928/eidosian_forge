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
def test_node_no_arg(self) -> None:
    self.assertTrue(defs.has('Relu'))
    node_def = helper.make_node('Relu', ['X'], ['Y'], name='test')
    self.assertEqual(node_def.op_type, 'Relu')
    self.assertEqual(node_def.name, 'test')
    self.assertEqual(list(node_def.input), ['X'])
    self.assertEqual(list(node_def.output), ['Y'])