from __future__ import annotations
import os
import typing
from typing import IO, Literal, Union
from onnx import serialization
from onnx.onnx_cpp2py_export import ONNX_ML
from onnx.external_data_helper import (
from onnx.onnx_pb import (
from onnx.onnx_operators_pb import OperatorProto, OperatorSetProto
from onnx.onnx_data_pb import MapProto, OptionalProto, SequenceProto
from onnx.version import version as __version__
from onnx import (
def load_tensor_from_string(s: bytes, format: _SupportedFormat=_DEFAULT_FORMAT) -> TensorProto:
    """Loads a binary string (bytes) that contains serialized TensorProto.

    Args:
        s: a string, which contains serialized TensorProto
        format: The serialization format. When it is not specified, it is inferred
            from the file extension when ``f`` is a path. If not specified _and_
            ``f`` is not a path, 'protobuf' is used. The encoding is assumed to
            be "utf-8" when the format is a text format.

    Returns:
        Loaded in-memory TensorProto.
    """
    return _get_serializer(format).deserialize_proto(s, TensorProto())