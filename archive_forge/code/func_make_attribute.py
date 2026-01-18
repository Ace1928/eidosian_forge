import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def make_attribute(key: str, value: Any, doc_string: Optional[str]=None, attr_type: Optional[int]=None) -> AttributeProto:
    """Makes an AttributeProto based on the value type."""
    attr = AttributeProto()
    attr.name = key
    if doc_string:
        attr.doc_string = doc_string
    if isinstance(value, numbers.Integral):
        attr.i = int(value)
        attr.type = AttributeProto.INT
    elif isinstance(value, numbers.Real):
        attr.f = float(value)
        attr.type = AttributeProto.FLOAT
    elif isinstance(value, (str, bytes)):
        attr.s = _to_bytes(value)
        attr.type = AttributeProto.STRING
    elif isinstance(value, TensorProto):
        attr.t.CopyFrom(value)
        attr.type = AttributeProto.TENSOR
    elif isinstance(value, SparseTensorProto):
        attr.sparse_tensor.CopyFrom(value)
        attr.type = AttributeProto.SPARSE_TENSOR
    elif isinstance(value, GraphProto):
        attr.g.CopyFrom(value)
        attr.type = AttributeProto.GRAPH
    elif isinstance(value, TypeProto):
        attr.tp.CopyFrom(value)
        attr.type = AttributeProto.TYPE_PROTO
    elif isinstance(value, collections.abc.Iterable):
        value = list(value)
        if len(value) == 0 and attr_type is None:
            raise ValueError(f'Could not infer attribute `{key}` type from empty iterator')
        if attr_type is None:
            types = {type(v) for v in value}
            for exp_t, exp_enum in ((numbers.Integral, AttributeProto.INTS), (numbers.Real, AttributeProto.FLOATS), ((str, bytes), AttributeProto.STRINGS), (TensorProto, AttributeProto.TENSORS), (SparseTensorProto, AttributeProto.SPARSE_TENSORS), (GraphProto, AttributeProto.GRAPHS), (TypeProto, AttributeProto.TYPE_PROTOS)):
                if all((issubclass(t, exp_t) for t in types)):
                    attr_type = exp_enum
                    break
            if attr_type is None:
                raise ValueError('Could not infer the attribute type from the elements of the passed Iterable value.')
        if attr_type == AttributeProto.INTS:
            attr.ints.extend(value)
            attr.type = AttributeProto.INTS
        elif attr_type == AttributeProto.FLOATS:
            attr.floats.extend(value)
            attr.type = AttributeProto.FLOATS
        elif attr_type == AttributeProto.STRINGS:
            attr.strings.extend((_to_bytes(v) for v in value))
            attr.type = AttributeProto.STRINGS
        elif attr_type == AttributeProto.TENSORS:
            attr.tensors.extend(value)
            attr.type = AttributeProto.TENSORS
        elif attr_type == AttributeProto.SPARSE_TENSORS:
            attr.sparse_tensors.extend(value)
            attr.type = AttributeProto.SPARSE_TENSORS
        elif attr_type == AttributeProto.GRAPHS:
            attr.graphs.extend(value)
            attr.type = AttributeProto.GRAPHS
        elif attr_type == AttributeProto.TYPE_PROTOS:
            attr.type_protos.extend(value)
            attr.type = AttributeProto.TYPE_PROTOS
        else:
            raise AssertionError()
    else:
        raise TypeError(f"'{value}' is not an accepted attribute value.")
    if attr_type is not None and attr.type != attr_type:
        raise TypeError(f"Inferred attribute type '{_attr_type_to_str(attr.type)}'({attr.type}) mismatched with specified type '{_attr_type_to_str(attr_type)}'({attr_type})")
    return attr