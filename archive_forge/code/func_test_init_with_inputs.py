import contextlib
import unittest
from typing import List, Sequence
import parameterized
import onnx
from onnx import defs
def test_init_with_inputs(self) -> None:
    op_schema = defs.OpSchema('test_op', 'test_domain', 1, inputs=[defs.OpSchema.FormalParameter('input1', 'T')], type_constraints=[('T', ['tensor(int64)'], '')])
    self.assertEqual(op_schema.name, 'test_op')
    self.assertEqual(op_schema.domain, 'test_domain')
    self.assertEqual(op_schema.since_version, 1)
    self.assertEqual(len(op_schema.inputs), 1)
    self.assertEqual(op_schema.inputs[0].name, 'input1')
    self.assertEqual(op_schema.inputs[0].type_str, 'T')
    self.assertEqual(len(op_schema.type_constraints), 1)
    self.assertEqual(op_schema.type_constraints[0].type_param_str, 'T')
    self.assertEqual(op_schema.type_constraints[0].allowed_type_strs, ['tensor(int64)'])