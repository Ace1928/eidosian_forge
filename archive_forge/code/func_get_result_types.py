from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from onnx import load
from onnx.defs import onnx_opset_version
from onnx.external_data_helper import ExternalDataInfo, uses_external_data
from onnx.model_container import ModelContainer
from onnx.onnx_pb import (
from onnx.reference.op_run import (
from onnx.reference.ops_optimized import optimized_operators
def get_result_types(self, name: str, exc: bool=True) -> Any:
    if self.all_types_ is None:
        raise RuntimeError(f'Unable to return type for name {name!r}. Run shape_inference first.')
    if name not in self.all_types_:
        if exc:
            raise RuntimeError(f'Unable to return type for name {name!r}, it was not found in {sorted(self.all_types_)}.')
        return None
    return self.all_types_[name]