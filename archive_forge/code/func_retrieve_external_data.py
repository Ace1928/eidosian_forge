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
def retrieve_external_data(self, initializer: TensorProto) -> np.array:
    """Returns a tensor saved as external."""
    info = ExternalDataInfo(initializer)
    location = info.location
    if self.container_ and self.container_.is_in_memory_external_initializer(location):
        return self.container_[location]
    if self.container_ is not None:
        raise RuntimeError('ReferenceEvaluator assumes a LargeContainer was loaded with its external tensor.')
    raise RuntimeError('An instance of LargeContainer should be created before using ReferenceEvaluator.')