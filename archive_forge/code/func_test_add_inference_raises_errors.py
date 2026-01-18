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
def test_add_inference_raises_errors(self) -> None:
    with self.assertRaises(ValidationError):
        _run_case(ADD_SCHEMA, ['A'], ['C'], _to_tensor_types({'A': (TensorProto.FLOAT, (3, 4))}))
    with self.assertRaises(ValidationError):
        _run_case(ADD_SCHEMA, ['A', 'B'], ['C'], _to_tensor_types({'A': (TensorProto.FLOAT, (3, 4)), 'B': (2, (3, 4))}))
    with self.assertRaises(InferenceError):
        _run_case(ADD_SCHEMA, ['A', 'B'], ['C'], _to_tensor_types({'A': (TensorProto.FLOAT, (2, 4)), 'B': (TensorProto.FLOAT, (3, 4))}))
    with self.assertRaises(KeyError):
        _run_case(ADD_SCHEMA, ['A', 'B'], ['C'], _to_tensor_types({'A': (TensorProto.FLOAT, (3, 4))}))