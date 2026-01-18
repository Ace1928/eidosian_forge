from collections import namedtuple
from typing import Any, Dict, NewType, Optional, Sequence, Tuple, Type
import numpy
import onnx.checker
import onnx.onnx_cpp2py_export.checker as c_checker
from onnx import IR_VERSION, ModelProto, NodeProto
@classmethod
def run_model(cls, model: ModelProto, inputs: Any, device: str='CPU', **kwargs: Any) -> Tuple[Any, ...]:
    backend = cls.prepare(model, device, **kwargs)
    assert backend is not None
    return backend.run(inputs)