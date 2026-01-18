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
def test_load_and_save_model_without_specifying_extension_or_format_defaults_to_protobuf(self) -> None:
    proto = _simple_model()
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, 'model')
        onnx.save_model(proto, model_path)
        with self.assertRaises(google.protobuf.text_format.ParseError):
            onnx.load_model(model_path, format='textproto')
        loaded_proto = onnx.load_model(model_path)
        self.assertEqual(proto, loaded_proto)
        loaded_proto_as_explicitly_protobuf = onnx.load_model(model_path, format='protobuf')
        self.assertEqual(proto, loaded_proto_as_explicitly_protobuf)