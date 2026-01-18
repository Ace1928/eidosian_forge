from pathlib import Path
from typing import List, Tuple, Union
import onnx
from onnx.external_data_helper import ExternalDataInfo, _get_initializer_tensors
def check_model_uses_external_data(model: onnx.ModelProto) -> bool:
    """
    Checks if the model uses external data.
    """
    model_tensors = _get_initializer_tensors(model)
    return any((tensor.HasField('data_location') and tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model_tensors))