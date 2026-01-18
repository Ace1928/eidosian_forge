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
def test_model_irversion(self) -> None:

    def mk_model(opset_versions: List[Tuple[str, int]]) -> ModelProto:
        graph = helper.make_graph([], 'my graph', [], [])
        return helper.make_model_gen_version(graph, opset_imports=[helper.make_opsetid(*pair) for pair in opset_versions])

    def test(opset_versions: List[Tuple[str, int]], ir_version: int) -> None:
        model = mk_model(opset_versions)
        self.assertEqual(model.ir_version, ir_version)
    test([('', 9)], 4)
    test([('', 10)], 5)
    test([('', 11)], 6)
    test([('', 12)], 7)
    test([('', 13)], 7)
    test([('', 14)], 7)
    test([('', 15)], 8)
    test([('', 16)], 8)
    test([('', 17)], 8)
    test([('', 18)], 8)
    test([('', 19)], 9)
    test([('', 20)], 9)
    test([('', 21)], 10)
    test([('ai.onnx', 9)], 4)
    test([('ai.onnx.ml', 2)], 6)
    test([('ai.onnx.ml', 3)], 8)
    test([('ai.onnx.ml', 4)], 9)
    test([('ai.onnx.ml', 5)], 10)
    test([('ai.onnx.training', 1)], 7)
    test([('', 10), ('ai.onnx.ml', 2)], 6)
    self.assertRaises(ValueError, mk_model, [('', 100)])