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
def test_large_onnx_no_large_initializer(self):
    model_proto = _linear_regression()
    large_model = onnx.model_container.make_large_model(model_proto.graph)
    self.common_check_reference_evaluator(large_model)
    with self.assertRaises(ValueError):
        large_model['#anymissingkey']
    with tempfile.TemporaryDirectory() as temp:
        filename = os.path.join(temp, 'model.onnx')
        large_model.save(filename)
        copy = onnx.model_container.ModelContainer()
        copy.load(filename)
        self.common_check_reference_evaluator(copy)