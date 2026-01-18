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
def test_infer_with_initializer_without_input_above_ir4(self) -> None:
    initializer_shape = (8, 7)
    original_model = self.prepare_input_initializer_tensors(initializer_shape, None)
    inferred_model = onnx.shape_inference.infer_shapes(original_model, strict_mode=True)
    z_tenor = inferred_model.graph.value_info.pop()
    z_shape = (z_tenor.type.tensor_type.shape.dim[0].dim_value, z_tenor.type.tensor_type.shape.dim[1].dim_value)
    assert z_shape == initializer_shape