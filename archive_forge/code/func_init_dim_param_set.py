from typing import Any, Dict, List, Set
import onnx.checker
from onnx import ModelProto, ValueInfoProto
def init_dim_param_set(dim_param_set: Set[str], value_infos: List[ValueInfoProto]) -> None:
    for info in value_infos:
        shape = info.type.tensor_type.shape
        for dim in shape.dim:
            if dim.HasField('dim_param'):
                dim_param_set.add(dim.dim_param)