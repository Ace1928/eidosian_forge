import io
import os
import pathlib
import tempfile
import unittest
import google.protobuf.message
import google.protobuf.text_format
import parameterized
import onnx
from onnx import serialization
def test_load_model_when_input_is_bytes(self) -> None:
    proto = _simple_model()
    proto_string = serialization.registry.get(self.format).serialize_proto(proto)
    loaded_proto = onnx.load_model_from_string(proto_string, format=self.format)
    self.assertEqual(proto, loaded_proto)