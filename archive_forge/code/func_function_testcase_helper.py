import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import onnx
from onnx.backend.test.case.test_case import TestCase
from onnx.backend.test.case.utils import import_recursive
from onnx.onnx_pb import (
def function_testcase_helper(node: NodeProto, input_types: List[TypeProto], name: str) -> Tuple[List[Tuple[List[NodeProto], Any]], int]:
    test_op = node.op_type
    op_prefix = test_op + '_' + name + '_expanded_function_'
    schema = onnx.defs.get_schema(test_op, domain=node.domain)
    function_protos = []
    for opset_version in schema.function_opset_versions:
        function_proto_str = schema.get_function_with_opset_version(opset_version)
        function_proto = FunctionProto()
        function_proto.ParseFromString(function_proto_str)
        function_protos.append(function_proto)
    for opset_version in schema.context_dependent_function_opset_versions:
        function_proto_str = schema.get_context_dependent_function_with_opset_version(opset_version, node.SerializeToString(), [t.SerializeToString() for t in input_types])
        function_proto = FunctionProto()
        function_proto.ParseFromString(function_proto_str)
        function_protos.append(function_proto)
    expanded_tests = []
    for function_proto in function_protos:
        for attr in schema.attributes:
            if attr in [a.name for a in node.attribute]:
                continue
            if schema.attributes[attr].default_value:
                node.attribute.extend([schema.attributes[attr].default_value])
        node_list = function_expand_helper(node, function_proto, op_prefix)
        expanded_tests.append((node_list, function_proto.opset_import))
    return (expanded_tests, schema.since_version)