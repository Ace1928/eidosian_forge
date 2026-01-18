import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from onnx import MapProto, OptionalProto, SequenceProto, TensorProto, helper, subbyte
from onnx.external_data_helper import load_external_data_for_tensor, uses_external_data
def from_optional(opt: Optional[Any], name: Optional[str]=None, dtype: Optional[int]=None) -> OptionalProto:
    """Converts an optional value into a Optional def.

    Args:
        opt: a Python optional
        name: (optional) the name of the optional.
        dtype: (optional) type of element in the input, used for specifying
                          optional values when converting empty none. dtype must
                          be a valid OptionalProto.DataType value

    Returns:
        optional: the converted optional def.
    """
    optional = OptionalProto()
    if name:
        optional.name = name
    if dtype:
        valid_dtypes = list(OptionalProto.DataType.values())
        if dtype not in valid_dtypes:
            raise TypeError(f'{dtype} must be a valid OptionalProto.DataType.')
        elem_type = dtype
    elif isinstance(opt, dict):
        elem_type = OptionalProto.MAP
    elif isinstance(opt, list):
        elem_type = OptionalProto.SEQUENCE
    elif opt is None:
        elem_type = OptionalProto.UNDEFINED
    else:
        elem_type = OptionalProto.TENSOR
    optional.elem_type = elem_type
    if opt is not None:
        if elem_type == OptionalProto.TENSOR:
            optional.tensor_value.CopyFrom(from_array(opt))
        elif elem_type == OptionalProto.SEQUENCE:
            optional.sequence_value.CopyFrom(from_list(opt))
        elif elem_type == OptionalProto.MAP:
            optional.map_value.CopyFrom(from_dict(opt))
        else:
            raise TypeError('The element type in the input is not a tensor, sequence, or map and is not supported.')
    return optional