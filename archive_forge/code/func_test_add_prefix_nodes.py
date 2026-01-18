import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def test_add_prefix_nodes(self) -> None:
    """Tests renaming nodes only"""
    self._test_add_prefix(rename_nodes=True)