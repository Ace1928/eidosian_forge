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
@contextlib.contextmanager
def enable_fake_mode():
    """Enable fake mode for the duration of the context.

    Internally it instantiates a :class:`torch._subclasses.fake_tensor.FakeTensorMode` context manager
    that converts user input and model parameters into :class:`torch._subclasses.fake_tensor.FakeTensor`.

    A :class:`torch._subclasses.fake_tensor.FakeTensor`
    is a :class:`torch.Tensor` with the ability to run PyTorch code without having to
    actually do computation through tensors allocated on a ``meta`` device. Because
    there is no actual data being allocated on the device, this API allows for
    exporting large models without the actual memory footprint needed for executing it.

    It is highly recommended to enable fake mode when exporting models that
    are too large to fit into memory.

    Returns:
        A :class:`ONNXFakeContext` object that must be passed to :func:`dynamo_export`
        through the :attr:`ExportOptions.fake_context` argument.

    Example::

        # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
        >>> import torch
        >>> import torch.onnx
        >>> class MyModel(torch.nn.Module):  # Dummy model
        ...     def __init__(self) -> None:
        ...         super().__init__()
        ...         self.linear = torch.nn.Linear(2, 2)
        ...     def forward(self, x):
        ...         out = self.linear(x)
        ...         return out
        >>> with torch.onnx.enable_fake_mode() as fake_context:
        ...     my_nn_module = MyModel()
        ...     arg1 = torch.randn(2, 2, 2)  # positional input 1
        >>> export_options = torch.onnx.ExportOptions(fake_context=fake_context)
        >>> onnx_program = torch.onnx.dynamo_export(
        ...     my_nn_module,
        ...     arg1,
        ...     export_options=export_options
        ... )
        >>> # Saving model WITHOUT initializers
        >>> onnx_program.save("my_model_without_initializers.onnx")
        >>> # Saving model WITH initializers
        >>> onnx_program.save("my_model_with_initializers.onnx", model_state_dict=MyModel().state_dict())

    .. warning::
        This API is experimental and is *NOT* backward-compatible.

    """
    from torch._subclasses import fake_tensor
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    fake_mode = fake_tensor.FakeTensorMode(allow_non_fake_inputs=not torch._guards.detect_fake_mode(), shape_env=ShapeEnv(allow_scalar_outputs=False, allow_dynamic_output_shape_ops=False))
    patcher_context = patcher.ONNXTorchPatcher()
    fake_context = ONNXFakeContext(fake_mode=fake_mode)
    with fake_mode, patcher_context:
        yield fake_context
    fake_context.state_dict_paths = tuple(patcher_context.paths)