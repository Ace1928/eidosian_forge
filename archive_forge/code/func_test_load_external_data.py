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
def test_load_external_data(self) -> None:
    model = onnx.load_model(self.model_filename, self.serialization_format)
    initializer_tensor = model.graph.initializer[0]
    np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)
    attribute_tensor = model.graph.node[0].attribute[0].t
    np.testing.assert_allclose(to_array(attribute_tensor), self.attribute_value)