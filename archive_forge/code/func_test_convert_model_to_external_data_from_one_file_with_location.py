from __future__ import annotations
import itertools
import os
import pathlib
import tempfile
import unittest
import uuid
from typing import Any
import numpy as np
import parameterized
import onnx
from onnx import ModelProto, TensorProto, checker, helper, shape_inference
from onnx.external_data_helper import (
from onnx.numpy_helper import from_array, to_array
def test_convert_model_to_external_data_from_one_file_with_location(self) -> None:
    model_file_path = self.get_temp_model_filename()
    external_data_file = str(uuid.uuid4())
    convert_model_to_external_data(self.model, size_threshold=0, all_tensors_to_one_file=True, location=external_data_file)
    onnx.save_model(self.model, model_file_path, self.serialization_format)
    self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, external_data_file)))
    model = onnx.load_model(model_file_path, self.serialization_format)
    convert_model_from_external_data(model)
    model_file_path = self.get_temp_model_filename()
    onnx.save_model(model, model_file_path, self.serialization_format)
    model = onnx.load_model(model_file_path, self.serialization_format)
    initializer_tensor = model.graph.initializer[0]
    self.assertFalse(len(initializer_tensor.external_data))
    self.assertEqual(initializer_tensor.data_location, TensorProto.DEFAULT)
    np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)
    attribute_tensor = model.graph.node[0].attribute[0].t
    self.assertFalse(len(attribute_tensor.external_data))
    self.assertEqual(attribute_tensor.data_location, TensorProto.DEFAULT)
    np.testing.assert_allclose(to_array(attribute_tensor), self.attribute_value)