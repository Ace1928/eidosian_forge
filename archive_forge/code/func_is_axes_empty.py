from typing import Any, Dict
import numpy as np
from onnx.onnx_pb import NodeProto
from onnx.reference.op_run import OpRun, RuntimeTypeError
def is_axes_empty(self, axes):
    return axes is None