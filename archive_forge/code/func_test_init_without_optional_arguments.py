import contextlib
import unittest
from typing import List, Sequence
import parameterized
import onnx
from onnx import defs
def test_init_without_optional_arguments(self) -> None:
    op_schema = defs.OpSchema('test_op', 'test_domain', 1)
    self.assertEqual(op_schema.name, 'test_op')
    self.assertEqual(op_schema.domain, 'test_domain')
    self.assertEqual(op_schema.since_version, 1)
    self.assertEqual(len(op_schema.inputs), 0)
    self.assertEqual(len(op_schema.outputs), 0)
    self.assertEqual(len(op_schema.type_constraints), 0)