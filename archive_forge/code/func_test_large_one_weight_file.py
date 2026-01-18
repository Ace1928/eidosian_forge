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
def test_large_one_weight_file(self):
    large_model = _large_linear_regression()
    self.common_check_reference_evaluator(large_model)
    with tempfile.TemporaryDirectory() as temp:
        filename = os.path.join(temp, 'model.onnx')
        large_model.save(filename, True)
        copy = onnx.model_container.ModelContainer()
        copy.load(filename)
        loaded_model = onnx.load_model(filename, load_external_data=True)
        self.common_check_reference_evaluator(loaded_model)