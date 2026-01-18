import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def test_case_connect_partially_no_name_collision(self) -> None:
    """Tests a scenario where two models without overlapping names are merged by
        connecting some outputs from the first model to some inputs in the second.
        The remaining inputs/outputs should be present in the combined model
        """

    def check_expectations(g1: GraphProto, g2: GraphProto, g4: GraphProto) -> None:
        del g1, g2
        self.assertEqual(['A0', 'A1', '_A', 'B21'], [elem.name for elem in g4.input])
        self.assertEqual(['B20', 'D0'], [elem.name for elem in g4.output])
    io_map = [('B00', 'B01'), ('B10', 'B11')]
    self._test_merge_models(M1_DEF, M2_DEF, io_map, check_expectations)