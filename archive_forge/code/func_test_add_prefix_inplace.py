import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def test_add_prefix_inplace(self) -> None:
    """Tests prefixing inplace"""
    self._test_add_prefix(inplace=True)