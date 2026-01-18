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
def test_check_model_relative(self) -> None:
    """More relative path test."""
    self.model_filename = self.create_test_model('..\\test\\..\\file.bin')
    with self.assertRaises(onnx.checker.ValidationError):
        checker.check_model(self.model_filename)