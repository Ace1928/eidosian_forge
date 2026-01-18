import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def test_add_prefix_edges(self) -> None:
    """Tests prefixing nodes edges. This will also rename inputs/outputs, since the names are shared"""
    self._test_add_prefix(rename_edges=True)