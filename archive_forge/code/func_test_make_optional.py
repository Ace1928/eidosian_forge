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
def test_make_optional(self) -> None:
    values = [1.1, 2.2, 3.3, 4.4, 5.5]
    values_tensor = helper.make_tensor(name='test', data_type=TensorProto.FLOAT, dims=(5,), vals=values)
    optional = helper.make_optional(name='test', elem_type=OptionalProto.TENSOR, value=values_tensor)
    self.assertEqual(optional.name, 'test')
    self.assertEqual(optional.elem_type, OptionalProto.TENSOR)
    self.assertEqual(optional.tensor_value, values_tensor)
    values_sequence = helper.make_sequence(name='test', elem_type=SequenceProto.TENSOR, values=[values_tensor, values_tensor])
    optional = helper.make_optional(name='test', elem_type=OptionalProto.SEQUENCE, value=values_sequence)
    self.assertEqual(optional.name, 'test')
    self.assertEqual(optional.elem_type, OptionalProto.SEQUENCE)
    self.assertEqual(optional.sequence_value, values_sequence)
    optional_none = helper.make_optional(name='test', elem_type=OptionalProto.UNDEFINED, value=None)
    self.assertEqual(optional_none.name, 'test')
    self.assertEqual(optional_none.elem_type, OptionalProto.UNDEFINED)
    self.assertFalse(optional_none.HasField('tensor_value'))