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
def test_adam_multiple(self) -> None:
    graph = self._make_graph([('R', TensorProto.FLOAT, ()), ('T', TensorProto.INT64, ()), ('X1', TensorProto.FLOAT, (1, 2)), ('X2', TensorProto.FLOAT, (3, 4)), ('G1', TensorProto.FLOAT, (1, 2)), ('G2', TensorProto.FLOAT, (3, 4)), ('V1', TensorProto.FLOAT, (1, 2)), ('V2', TensorProto.FLOAT, (3, 4)), ('H1', TensorProto.FLOAT, (1, 2)), ('H2', TensorProto.FLOAT, (3, 4))], [make_node('Adam', ['R', 'T', 'X1', 'X2', 'G1', 'G2', 'V1', 'V2', 'H1', 'H2'], ['X1_new', 'X2_new', 'V1_new', 'V2_new', 'H1_new', 'H2_new'], domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN, alpha=0.9, beta=1.0, norm_coefficient=0.02)], [])
    infos = [make_tensor_value_info('X1_new', TensorProto.FLOAT, (1, 2)), make_tensor_value_info('X2_new', TensorProto.FLOAT, (3, 4)), make_tensor_value_info('V1_new', TensorProto.FLOAT, (1, 2)), make_tensor_value_info('V2_new', TensorProto.FLOAT, (3, 4)), make_tensor_value_info('H1_new', TensorProto.FLOAT, (1, 2)), make_tensor_value_info('H2_new', TensorProto.FLOAT, (3, 4))]
    self._assert_inferred(graph, infos, opset_imports=[make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1), make_opsetid(ONNX_DOMAIN, 12)])