import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_MelWeightMatrix(self) -> None:
    num_mel_bins = helper.make_tensor('a', TensorProto.INT64, dims=[], vals=np.array([10]))
    dft_length = helper.make_tensor('b', TensorProto.INT64, dims=[], vals=np.array([64]))
    sample_rate = helper.make_tensor('c', TensorProto.INT64, dims=[], vals=np.array([0]))
    lower_edge_hertz = helper.make_tensor('d', TensorProto.FLOAT, dims=[], vals=np.array([0]))
    upper_edge_hertz = helper.make_tensor('e', TensorProto.FLOAT, dims=[], vals=np.array([1]))
    self._test_op_upgrade('MelWeightMatrix', 17, [[], [], [], [], []], [[33, 10]], [TensorProto.INT64, TensorProto.INT64, TensorProto.INT64, TensorProto.FLOAT, TensorProto.FLOAT], initializer=[num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz])
    num_mel_bins = helper.make_tensor('a', TensorProto.INT64, dims=[], vals=np.array([20]))
    dft_length = helper.make_tensor('b', TensorProto.INT64, dims=[], vals=np.array([31]))
    sample_rate = helper.make_tensor('c', TensorProto.INT64, dims=[], vals=np.array([0]))
    lower_edge_hertz = helper.make_tensor('d', TensorProto.FLOAT, dims=[], vals=np.array([0]))
    upper_edge_hertz = helper.make_tensor('e', TensorProto.FLOAT, dims=[], vals=np.array([1]))
    self._test_op_upgrade('MelWeightMatrix', 17, [[], [], [], [], []], [[16, 20]], [TensorProto.INT64, TensorProto.INT64, TensorProto.INT64, TensorProto.FLOAT, TensorProto.FLOAT], initializer=[num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz])