import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def test_case_connect_all_no_name_collision(self) -> None:
    """Tests a simple scenario where two models without overlapping names are merged by
        connecting all the outputs in the first models to all the inputs in the second model
        """

    def check_expectations(g1: GraphProto, g2: GraphProto, g3: GraphProto) -> None:
        self.assertEqual(g3.input, g1.input)
        self.assertEqual(g3.output, g2.output)
        self.assertEqual(['Add', 'Sub', 'Mul', 'Add', 'Sub', 'Mul'], [item.op_type for item in g3.node])
    io_map = [('B00', 'B01'), ('B10', 'B11'), ('B20', 'B21')]
    self._test_merge_models(M1_DEF, M2_DEF, io_map, check_expectations)