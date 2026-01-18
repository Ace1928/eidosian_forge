from collections import namedtuple
from typing import Any, Dict, NewType, Optional, Sequence, Tuple, Type
import numpy
import onnx.checker
import onnx.onnx_cpp2py_export.checker as c_checker
from onnx import IR_VERSION, ModelProto, NodeProto
@classmethod
def supports_device(cls, device: str) -> bool:
    """Checks whether the backend is compiled with particular device support.
        In particular it's used in the testing suite.
        """
    return True