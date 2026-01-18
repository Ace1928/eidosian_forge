import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def test_add_prefix_outputs(self) -> None:
    """Tests prefixing graph outputs only. Relevant node edges should be renamed as well"""
    self._test_add_prefix(rename_outputs=True)