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
def save_tensor(proto: TensorProto, f: IO[bytes] | str | os.PathLike, format: _SupportedFormat | None=None) -> None:
    """Saves the TensorProto to the specified path.

    Args:
        proto: should be a in-memory TensorProto
        f: can be a file-like object (has "write" function) or a string
        containing a file name or a pathlike object.
        format: The serialization format. When it is not specified, it is inferred
            from the file extension when ``f`` is a path. If not specified _and_
            ``f`` is not a path, 'protobuf' is used. The encoding is assumed to
            be "utf-8" when the format is a text format.
    """
    serialized = _get_serializer(format, f).serialize_proto(proto)
    _save_bytes(serialized, f)