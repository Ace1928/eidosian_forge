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
@parameterized.expand([(name, version, test_aspect, input_shape, axis, onesided, inverse, expected_shape) for (name, version), (test_aspect, input_shape, axis, onesided, inverse, expected_shape) in itertools.product(all_versions_for('DFT'), (('reals_default_axis', (2, 5, 1), None, None, None, (2, 42, 2)), ('reals_axis_0', (3, 5, 10, 1), 0, 0, 0, (42, 5, 10, 2)), ('reals_axis_1', (3, 5, 10, 1), 1, 0, 0, (3, 42, 10, 2)), ('reals_axis_2', (3, 5, 10, 1), 2, 0, 0, (3, 5, 42, 2)), ('reals_axis_neg', (3, 5, 10, 1), -2, 0, 0, (3, 5, 42, 2)), ('reals_axis_0_onesided', (3, 5, 10, 1), 0, 1, 0, (22, 5, 10, 2)), ('reals_axis_1_onesided', (3, 5, 10, 1), 1, 1, 0, (3, 22, 10, 2)), ('reals_axis_2_onesided', (3, 5, 10, 1), 2, 1, 0, (3, 5, 22, 2)), ('reals_axis_neg_onesided', (3, 5, 10, 1), -2, 1, 0, (3, 5, 22, 2)), ('complex_default_axis', (2, 5, 2), None, None, None, (2, 42, 2)), ('complex_onesided', (2, 5, 2), 1, 1, None, (2, 22, 2)), ('real_inverse', (2, 5, 1), 1, None, 1, (2, 42, 2)), ('complex_inverse', (2, 5, 2), 1, None, 1, (2, 42, 2))))])
def test_dft_dft_length(self, _: str, version: int, _test_aspect: str, input_shape: tuple[int], axis: int | None, onesided: int | None, inverse: int | None, expected_shape: tuple[int]) -> None:
    attributes = {}
    if onesided is not None:
        attributes['onesided'] = onesided
    if inverse is not None:
        attributes['inverse'] = inverse
    dft_length = 42
    if version < 20:
        if axis is not None:
            attributes['axis'] = axis
        nodes = [make_node('Constant', [], ['dft_length'], value=make_tensor('dft_length', TensorProto.INT64, (), (dft_length,))), make_node('DFT', ['input', 'dft_length'], ['output'], **attributes)]
        value_infos = [make_tensor_value_info('dft_length', TensorProto.INT64, ())]
    else:
        assert version >= 20
        if axis is not None:
            nodes = [make_node('Constant', [], ['axis'], value=make_tensor('axis', TensorProto.INT64, (), (axis,))), make_node('Constant', [], ['dft_length'], value=make_tensor('dft_length', TensorProto.INT64, (), (dft_length,))), make_node('DFT', ['input', 'dft_length', 'axis'], ['output'], **attributes)]
            value_infos = [make_tensor_value_info('dft_length', TensorProto.INT64, ()), make_tensor_value_info('axis', TensorProto.INT64, ())]
        else:
            nodes = [make_node('Constant', [], ['dft_length'], value=make_tensor('dft_length', TensorProto.INT64, (), (dft_length,))), make_node('DFT', ['input', 'dft_length', ''], ['output'], **attributes)]
            value_infos = [make_tensor_value_info('dft_length', TensorProto.INT64, ())]
    graph = self._make_graph([], [make_node('Constant', [], ['input'], value=make_tensor('input', TensorProto.FLOAT, input_shape, np.ones(input_shape, dtype=np.float32).flatten())), *nodes], [])
    self._assert_inferred(graph, [make_tensor_value_info('input', TensorProto.FLOAT, input_shape), *value_infos, make_tensor_value_info('output', TensorProto.FLOAT, expected_shape)], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)])