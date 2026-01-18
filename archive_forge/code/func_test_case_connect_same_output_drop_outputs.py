import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def test_case_connect_same_output_drop_outputs(self) -> None:
    """Tests a scenario where we merge two models by connecting a single output in the first model
        to all the inputs in the second, while dropping the rest of the outputs in the first model
        """

    def check_expectations(g1: GraphProto, g2: GraphProto, g3: GraphProto) -> None:
        del g2
        self.assertEqual(g3.input, g1.input)
        self.assertEqual(['D0'], [elem.name for elem in g3.output])
        self.assertEqual(['Add', 'Add', 'Sub', 'Mul'], [item.op_type for item in g3.node])
    io_map = [('B00', 'B01'), ('B00', 'B11'), ('B00', 'B21')]
    outputs = ['D0']
    self._test_merge_models(M1_DEF, M2_DEF, io_map, check_expectations, outputs=outputs)