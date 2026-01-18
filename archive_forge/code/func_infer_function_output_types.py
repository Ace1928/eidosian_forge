from __future__ import annotations
import os
from typing import Sequence
import onnx
import onnx.onnx_cpp2py_export.shape_inference as C  # noqa: N812
from onnx import AttributeProto, FunctionProto, ModelProto, TypeProto
def infer_function_output_types(function: FunctionProto, input_types: Sequence[TypeProto], attributes: Sequence[AttributeProto]) -> list[TypeProto]:
    """Apply type-and-shape-inference to given function body, with given input types
    and given input attribute values.
    """
    result = C.infer_function_output_types(function.SerializeToString(), [x.SerializeToString() for x in input_types], [x.SerializeToString() for x in attributes])

    def to_type_proto(x) -> TypeProto:
        type_proto = onnx.TypeProto()
        type_proto.ParseFromString(x)
        return type_proto
    return [to_type_proto(x) for x in result]