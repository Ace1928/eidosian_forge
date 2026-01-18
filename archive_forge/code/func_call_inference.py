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
def call_inference():
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT32, (2, 4, 3, 9))], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)])