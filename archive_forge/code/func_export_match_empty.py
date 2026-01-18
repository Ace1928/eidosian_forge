import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_match_empty() -> None:
    node = onnx.helper.make_node('RegexFullMatch', inputs=['X'], outputs=['Y'], pattern='(\\W|^)[\\w.\\-]{0,25}@(yahoo|gmail)\\.com(\\W|$)')
    x = np.array([[], []]).astype(object)
    result = np.array([[], []]).astype(bool)
    expect(node, inputs=[x], outputs=[result], name='test_regex_full_match_empty')