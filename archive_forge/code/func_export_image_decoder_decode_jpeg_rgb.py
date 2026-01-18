from __future__ import annotations
import io
import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import _image_decoder_data, expect
@staticmethod
def export_image_decoder_decode_jpeg_rgb() -> None:
    node = onnx.helper.make_node('ImageDecoder', inputs=['data'], outputs=['output'], pixel_format='RGB')
    data, output = _generate_test_data('jpeg', _image_decoder_data.image_decoder_decode_jpeg_rgb, 'RGB')
    expect(node, inputs=[data], outputs=[output], name='test_image_decoder_decode_jpeg_rgb')