import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def make_tensor_sequence_value_info(name: str, elem_type: int, shape: Optional[Sequence[Union[str, int, None]]], doc_string: str='', elem_shape_denotation: Optional[List[str]]=None) -> ValueInfoProto:
    """Makes a Sequence[Tensors] ValueInfoProto based on the data type and shape."""
    value_info_proto = ValueInfoProto()
    value_info_proto.name = name
    if doc_string:
        value_info_proto.doc_string = doc_string
    tensor_type_proto = make_tensor_type_proto(elem_type, shape, elem_shape_denotation)
    sequence_type_proto = make_sequence_type_proto(tensor_type_proto)
    value_info_proto.type.sequence_type.CopyFrom(sequence_type_proto.sequence_type)
    return value_info_proto