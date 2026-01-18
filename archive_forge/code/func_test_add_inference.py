import unittest
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import onnx
from onnx import TensorProto, TypeProto
from onnx.checker import ValidationError
from onnx.defs import OpSchema, get_all_schemas_with_history, get_schema
from onnx.helper import (
from onnx.numpy_helper import from_array
from onnx.shape_inference import InferenceError, infer_node_outputs
def test_add_inference(self) -> None:
    cases = [({'A': (TensorProto.FLOAT, ()), 'B': (TensorProto.FLOAT, ())}, {'C': (TensorProto.FLOAT, ())}), ({'A': (TensorProto.FLOAT, (None, 2)), 'B': (TensorProto.FLOAT, (2,))}, {'C': (TensorProto.FLOAT, (None, 2))}), ({'A': (TensorProto.FLOAT, (None, 2)), 'B': (TensorProto.FLOAT, (1, 2))}, {'C': (TensorProto.FLOAT, (None, 2))}), ({'A': (TensorProto.DOUBLE, ('n', 'm')), 'B': (TensorProto.DOUBLE, (1, 'n', 'm'))}, {'C': (TensorProto.DOUBLE, (1, 'n', 'm'))}), ({'A': (TensorProto.FLOAT, ('x', 2)), 'B': (TensorProto.FLOAT, ('y', 2))}, {'C': (TensorProto.FLOAT, (None, 2))})]
    for ins, outs in cases:
        assert _run_case(ADD_SCHEMA, ['A', 'B'], ['C'], _to_tensor_types(ins)) == _to_tensor_types(outs)