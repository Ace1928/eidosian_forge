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
def test_load_model_picks_correct_format_from_extension(self) -> None:
    proto = _simple_model()
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, 'model.textproto')
        onnx.save_model(proto, model_path, format='textproto')
        loaded_proto = onnx.load_model(model_path)
        self.assertEqual(proto, loaded_proto)