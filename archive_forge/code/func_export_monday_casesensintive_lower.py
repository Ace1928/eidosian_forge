import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_monday_casesensintive_lower() -> None:
    input = np.array(['monday', 'tuesday', 'wednesday', 'thursday']).astype(object)
    output = np.array(['tuesday', 'wednesday', 'thursday']).astype(object)
    stopwords = ['monday']
    node = onnx.helper.make_node('StringNormalizer', inputs=['x'], outputs=['y'], case_change_action='LOWER', is_case_sensitive=1, stopwords=stopwords)
    expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_casesensintive_lower')