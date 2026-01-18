import itertools
import random
import struct
import unittest
from typing import Any, List, Tuple
import numpy as np
import parameterized
import pytest
import version_utils
from onnx import (
from onnx.reference.op_run import to_array_extended
def test_model_docstring(self) -> None:
    graph = helper.make_graph([], 'my graph', [], [])
    model_def = helper.make_model(graph, doc_string='test')
    self.assertFalse(hasattr(model_def, 'name'))
    self.assertEqual(model_def.doc_string, 'test')