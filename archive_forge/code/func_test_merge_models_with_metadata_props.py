import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def test_merge_models_with_metadata_props(self) -> None:
    m1 = _load_model(M1_DEF)
    helper.set_model_props(m1, {'p1': 'v1', 'p2': 'v2'})
    m2 = _load_model(M2_DEF)
    helper.set_model_props(m2, {'p3': 'v3', 'p4': 'v4'})
    io_map = [('B00', 'B01')]
    m3 = compose.merge_models(m1, m2, io_map=io_map)
    assert len(m3.metadata_props) == 4
    helper.set_model_props(m2, {'p1': 'v1', 'p4': 'v4'})
    m3 = compose.merge_models(m1, m2, io_map=io_map)
    assert len(m3.metadata_props) == 3
    helper.set_model_props(m2, {'p1': 'v5', 'p4': 'v4'})
    self.assertRaises(ValueError, compose.merge_models, m1, m2, io_map=io_map)