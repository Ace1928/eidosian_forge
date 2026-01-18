from pathlib import Path
from typing import List, Tuple, Union
import onnx
from onnx.external_data_helper import ExternalDataInfo, _get_initializer_tensors
def has_onnx_input(model: Union[onnx.ModelProto, Path, str], input_name: str) -> bool:
    """
    Checks if the model has a specific input.
    """
    if isinstance(model, (str, Path)):
        model = Path(model).as_posix()
        model = onnx.load(model, load_external_data=False)
    for input in model.graph.input:
        if input.name == input_name:
            return True
    return False