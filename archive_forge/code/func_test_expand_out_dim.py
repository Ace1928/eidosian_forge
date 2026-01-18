import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def test_expand_out_dim(self) -> None:
    """Tests expanding output dimensions. The resulting graph should have the same output names,
        but with one more dimension at the specified index.
        """
    m1 = _load_model(M1_DEF)

    def _check_model(m1: ModelProto, m2: ModelProto, dim_idx: int) -> None:
        for out_g2, out_g1 in zip(m2.graph.output, m1.graph.output):
            self.assertEqual(out_g2.name, out_g1.name)
            self.assertEqual(out_g2.type.tensor_type.elem_type, out_g1.type.tensor_type.elem_type)
            expected_out_shape = _get_shape(out_g1)
            expected_out_shape.insert(dim_idx, 1)
            self.assertEqual(_get_shape(out_g2), expected_out_shape)
    for dim_idx in [0, 2, -1, -3]:
        m2 = compose.expand_out_dim(m1, dim_idx)
        _check_model(m1, m2, dim_idx)
    m2 = ModelProto()
    m2.CopyFrom(m1)
    dim_idx = 0
    compose.expand_out_dim(m2, dim_idx, inplace=True)
    _check_model(m1, m2, dim_idx)