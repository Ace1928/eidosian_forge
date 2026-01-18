import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def test_error_wrong_input_output_name(self) -> None:
    """Tests that providing a non existing output/input name in the io_map argument produces an error."""
    m1, m2 = (_load_model(M1_DEF), _load_model(M2_DEF))
    self.assertRaises(ValueError, compose.merge_models, m1, m2, io_map=[('wrong_outname', 'B01'), ('B10', 'B11'), ('B20', 'B21')])
    self.assertRaises(ValueError, compose.merge_models, m1, m2, io_map=[('B00', 'wrong_input'), ('B10', 'B11'), ('B20', 'B21')])