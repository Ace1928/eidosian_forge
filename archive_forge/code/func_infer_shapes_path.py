from __future__ import annotations
import os
from typing import Sequence
import onnx
import onnx.onnx_cpp2py_export.shape_inference as C  # noqa: N812
from onnx import AttributeProto, FunctionProto, ModelProto, TypeProto
def infer_shapes_path(model_path: str | os.PathLike, output_path: str | os.PathLike='', check_type: bool=False, strict_mode: bool=False, data_prop: bool=False) -> None:
    """Take model path for shape_inference.

    This function is the same as :func:`infer_shape` but supports >2GB models.
    The function outputs the inferred model to the `output_path`. The original model path
    is used if not specified.
    """
    if isinstance(model_path, ModelProto):
        raise TypeError('infer_shapes_path only accepts model Path (String),you can use infer_shapes for the ModelProto.')
    try:
        model_path = os.fspath(model_path)
    except TypeError as exp:
        raise TypeError(f'infer_shapes_path only accepts model path as a string or PathLike, incorrect model path type: {type(model_path)}') from exp
    try:
        output_path = os.fspath(output_path)
    except TypeError as exp:
        raise TypeError(f'infer_shapes_path only accepts output path as a string or PathLike, incorrect output path type: {type(output_path)}') from exp
    if output_path == '':
        output_path = model_path
    C.infer_shapes_path(model_path, output_path, check_type, strict_mode, data_prop)