import unittest
from parameterized import parameterized
import onnx
from onnx import GraphProto, OperatorSetIdProto, checker
def expect_model_function_attribute(model):
    assert len(model.functions[0].attribute_proto) == 2
    attr_proto_alpha = [attr_proto for attr_proto in model.functions[0].attribute_proto if attr_proto.name == 'alpha']
    assert len(attr_proto_alpha) == 1 and attr_proto_alpha[0].f == default_alpha
    attr_proto_gamma = [attr_proto for attr_proto in model.functions[0].attribute_proto if attr_proto.name == 'gamma']
    assert len(attr_proto_gamma) == 1 and attr_proto_gamma[0].f == default_gamma