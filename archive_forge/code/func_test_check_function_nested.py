import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
def test_check_function_nested(self) -> None:
    func_domain = 'local'
    func_nested_opset_imports = [helper.make_opsetid('', 14), helper.make_opsetid(func_domain, 1)]
    func_nested_identity_add_name = 'func_nested_identity_add'
    func_nested_identity_add_inputs = ['a', 'b']
    func_nested_identity_add_outputs = ['c']
    func_nested_identity_add_nodes = [helper.make_node('func_identity', ['a'], ['a1'], domain=func_domain), helper.make_node('func_identity', ['b'], ['b1'], domain=func_domain), helper.make_node('func_add', ['a1', 'b1'], ['c'], domain=func_domain)]
    func_nested_identity_add = helper.make_function(func_domain, func_nested_identity_add_name, func_nested_identity_add_inputs, func_nested_identity_add_outputs, func_nested_identity_add_nodes, func_nested_opset_imports)
    checker.check_function(func_nested_identity_add)