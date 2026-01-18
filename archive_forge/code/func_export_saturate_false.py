import sys
import numpy as np
import onnx
import onnx.reference.custom_element_types as custom
from onnx import TensorProto, helper, subbyte
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.helper import (
from onnx.numpy_helper import float8e4m3_to_float32, float8e5m2_to_float32
@staticmethod
def export_saturate_false() -> None:
    test_cases = [('FLOAT', 'FLOAT8E4M3FN'), ('FLOAT16', 'FLOAT8E4M3FN'), ('FLOAT', 'FLOAT8E4M3FNUZ'), ('FLOAT16', 'FLOAT8E4M3FNUZ'), ('FLOAT', 'FLOAT8E5M2'), ('FLOAT16', 'FLOAT8E5M2'), ('FLOAT', 'FLOAT8E5M2FNUZ'), ('FLOAT16', 'FLOAT8E5M2FNUZ')]
    vect_float32_to_float8e4m3 = np.vectorize(float32_to_float8e4m3)
    vect_float32_to_float8e5m2 = np.vectorize(float32_to_float8e5m2)
    for from_type, to_type in test_cases:
        np_fp32 = np.array(['0.47892547', '0.48033667', '0.49968487', '0.81910545', '0.47031248', '0.7229038', '1000000', '1e-7', 'NaN', 'INF', '+INF', '-INF', '-0.0000001', '0.0000001', '-1000000'], dtype=np.float32)
        if from_type == 'FLOAT':
            input_values = np_fp32
            input = make_tensor('x', TensorProto.FLOAT, [3, 5], np_fp32.tolist())
        elif from_type == 'FLOAT16':
            input_values = np_fp32.astype(np.float16).astype(np.float32)
            input = make_tensor('x', TensorProto.FLOAT16, [3, 5], input_values.tolist())
        else:
            raise ValueError('Conversion from {from_type} to {to_type} is not tested.')
        if to_type == 'FLOAT8E4M3FN':
            expected = vect_float32_to_float8e4m3(input_values, saturate=False)
        elif to_type == 'FLOAT8E4M3FNUZ':
            expected = vect_float32_to_float8e4m3(input_values, uz=True, saturate=False)
        elif to_type == 'FLOAT8E5M2':
            expected = vect_float32_to_float8e5m2(input_values, saturate=False)
        elif to_type == 'FLOAT8E5M2FNUZ':
            expected = vect_float32_to_float8e5m2(input_values, fn=True, uz=True, saturate=False)
        else:
            raise ValueError('Conversion from {from_type} to {to_type} is not tested.')
        ivals = bytes([int(i) for i in expected])
        tensor = TensorProto()
        tensor.data_type = getattr(TensorProto, to_type)
        tensor.name = 'x'
        tensor.dims.extend([3, 5])
        field = tensor_dtype_to_field(tensor.data_type)
        getattr(tensor, field).extend(ivals)
        output = tensor
        node = onnx.helper.make_node('Cast', inputs=['input'], outputs=['output'], to=getattr(TensorProto, to_type), saturate=0)
        expect(node, inputs=[input], outputs=[output], name='test_cast_no_saturate_' + from_type + '_to_' + to_type)