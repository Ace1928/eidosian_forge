from __future__ import (  # for onnx.ModelProto (ONNXProgram) and onnxruntime (ONNXRuntimeOptions)
import abc
import contextlib
import dataclasses
import io
import logging
import os
import warnings
from collections import defaultdict
from typing import (
from typing_extensions import Self
import torch
import torch._ops
import torch.export as torch_export
import torch.utils._pytree as pytree
from torch._subclasses import fake_tensor
from torch.onnx._internal import _beartype, io_adapter
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.fx import (
@_beartype.beartype
def register_op(self, function: Union['onnxscript.OnnxFunction', 'onnxscript.TracedOnnxFunction'], namespace: str, op_name: str, overload: Optional[str]=None, is_complex: bool=False) -> None:
    """Registers a custom operator: torch.ops.<namespace>.<op_name>.<overload>.

        Args:
            function: The onnx-sctip function to register.
            namespace: The namespace of the operator to register.
            op_name: The name of the operator to register.
            overload: The overload of the operator to register. If it's default overload,
                leave it to None.
            is_complex: Whether the function is a function that handles complex valued inputs.

        Raises:
            ValueError: If the name is not in the form of 'namespace::op'.
        """
    internal_name_instance = registration.OpName.from_name_parts(namespace=namespace, op_name=op_name, overload=overload)
    symbolic_function = registration.ONNXFunction(onnx_function=function, op_full_name=internal_name_instance.qualified_name(), is_custom=True, is_complex=is_complex)
    self._register(internal_name_instance, symbolic_function)