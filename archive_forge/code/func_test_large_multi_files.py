import os
import tempfile
import unittest
import numpy as np
import numpy.testing as npt
import onnx
import onnx.helper
import onnx.model_container
import onnx.numpy_helper
import onnx.reference
def test_large_multi_files(self):
    large_model = _large_linear_regression()
    self.common_check_reference_evaluator(large_model)
    with tempfile.TemporaryDirectory() as temp:
        filename = os.path.join(temp, 'model.onnx')
        large_model.save(filename, False)
        copy = onnx.load_model(filename)
        self.common_check_reference_evaluator(copy)
        loaded_model = onnx.load_model(filename, load_external_data=True)
        self.common_check_reference_evaluator(loaded_model)