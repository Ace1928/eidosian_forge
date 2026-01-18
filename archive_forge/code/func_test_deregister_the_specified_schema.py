import contextlib
import unittest
from typing import List, Sequence
import parameterized
import onnx
from onnx import defs
def test_deregister_the_specified_schema(self):
    for version in [*self.trap_op_version, self.op_version]:
        op_schema = defs.OpSchema(self.op_type, self.op_domain, version)
        onnx.defs.register_schema(op_schema)
        self.assertTrue(onnx.defs.has(op_schema.name, version, op_schema.domain))
    onnx.defs.deregister_schema(op_schema.name, self.op_version, op_schema.domain)
    for version in self.trap_op_version:
        self.assertTrue(onnx.defs.has(op_schema.name, version, op_schema.domain))
    if onnx.defs.has(op_schema.name, self.op_version, op_schema.domain):
        schema = onnx.defs.get_schema(op_schema.name, self.op_version, op_schema.domain)
        self.assertLess(schema.since_version, self.op_version)