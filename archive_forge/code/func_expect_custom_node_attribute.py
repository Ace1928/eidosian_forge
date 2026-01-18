import unittest
from parameterized import parameterized
import onnx
from onnx import GraphProto, OperatorSetIdProto, checker
def expect_custom_node_attribute(node, attributes):
    for key in attributes:
        match_attr = [attr for attr in node.attribute if attr.name == key]
        assert len(match_attr) == 1
        assert match_attr[0].f == attributes[key]