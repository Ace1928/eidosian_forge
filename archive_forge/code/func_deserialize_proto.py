import os
import tempfile
import unittest
import onnx
def deserialize_proto(self, serialized: bytes, proto):
    text = serialized.decode('utf-8')
    if isinstance(proto, onnx.ModelProto):
        return onnx.parser.parse_model(text)
    if isinstance(proto, onnx.GraphProto):
        return onnx.parser.parse_graph(text)
    if isinstance(proto, onnx.FunctionProto):
        return onnx.parser.parse_function(text)
    if isinstance(proto, onnx.NodeProto):
        return onnx.parser.parse_node(text)
    raise ValueError(f'Unsupported proto type: {type(proto)}')