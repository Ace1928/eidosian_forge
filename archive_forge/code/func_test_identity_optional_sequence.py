from __future__ import annotations
import itertools
import unittest
from typing import Any, Sequence
import numpy as np
import pytest
from parameterized import parameterized
import onnx.shape_inference
from onnx import (
from onnx.defs import (
from onnx.helper import (
from onnx.parser import parse_graph
def test_identity_optional_sequence(self) -> None:
    graph = self._make_graph([('input1', TensorProto.FLOAT, (2, 3, 4)), ('input2', TensorProto.FLOAT, (2, 3, 4)), ('input3', TensorProto.FLOAT, (2, 5, 4))], [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['in_sequence']), make_node('Optional', ['in_sequence'], ['in_optional']), make_node('Identity', ['in_optional'], ['output_optional'])], [])
    tensor_type_proto = helper.make_tensor_type_proto(TensorProto.FLOAT, (2, None, 4))
    sequence_type_proto = helper.make_sequence_type_proto(tensor_type_proto)
    optional_type_proto = helper.make_optional_type_proto(sequence_type_proto)
    self._assert_inferred(graph, [helper.make_value_info('in_sequence', sequence_type_proto), helper.make_value_info('in_optional', optional_type_proto), helper.make_value_info('output_optional', optional_type_proto)])