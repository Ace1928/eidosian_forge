from __future__ import annotations
import os
import sys
from typing import Any, Callable, TypeVar
from google.protobuf.message import Message
import onnx.defs
import onnx.onnx_cpp2py_export.checker as C  # noqa: N812
import onnx.shape_inference
from onnx import (
def check_sparse_tensor(sparse: SparseTensorProto, ctx: C.CheckerContext=DEFAULT_CONTEXT) -> None:
    _ensure_proto_type(sparse, SparseTensorProto)
    C.check_sparse_tensor(sparse.SerializeToString(), ctx)