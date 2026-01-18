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
@parameterized.expand(all_versions_for('MatMul'))
def test_matmul_allow_unknown(self, _, version) -> None:
    self._make_matmul_test_allow_unknown(version, (None,), (None,), ())
    self._make_matmul_test_allow_unknown(version, (3,), (None,), ())
    self._make_matmul_test_allow_unknown(version, (2,), (2, 'a'), ('a',))
    self._make_matmul_test_allow_unknown(version, (4, 2), (2, 'a'), (4, 'a'))
    self._make_matmul_test_allow_unknown(version, (4, None), (2, 'a'), (4, 'a'))
    self._make_matmul_test_allow_unknown(version, (4, None), (None, 'a'), (4, 'a'))
    self._make_matmul_test_allow_unknown(version, (1, 4, 2), ('a', 2, 5), ('a', 4, 5))
    self._make_matmul_test_allow_unknown(version, (1, 3, 4, 2), ('a', 2, 5), (1, 3, 4, 5))
    self._make_matmul_test_allow_unknown(version, (3,), None, None)
    self._make_matmul_test_allow_unknown(version, None, None, None)