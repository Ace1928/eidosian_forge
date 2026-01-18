import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def check_expectations(g1: GraphProto, g2: GraphProto, g4: GraphProto) -> None:
    del g1, g2
    self.assertEqual(['A0', 'A1', '_A', 'B21'], [elem.name for elem in g4.input])
    self.assertEqual(['B20', 'D0'], [elem.name for elem in g4.output])