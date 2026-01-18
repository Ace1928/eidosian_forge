import contextlib
import unittest
from typing import List, Sequence
import parameterized
import onnx
from onnx import defs
def test_function_body(self) -> None:
    self.assertEqual(type(defs.get_schema('Selu').function_body), onnx.FunctionProto)