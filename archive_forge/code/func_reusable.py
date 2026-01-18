import dataclasses
import importlib
import logging
from typing import (
from typing_extensions import TypeAlias
import torch
import torch._C
import torch._ops
import torch._prims.executor
import torch.fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx._compatibility import compatibility
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.utils import _pytree
def reusable(a: OrtBackendOptions, b: OrtBackendOptions):
    if a.preferred_execution_providers != b.preferred_execution_providers or a.infer_execution_providers != b.infer_execution_providers or a.default_execution_providers != b.default_execution_providers or (a.preallocate_output != b.preallocate_output) or (a.use_aot_autograd != b.use_aot_autograd):
        return False
    if a.ort_session_options is not None or b.ort_session_options is not None:
        return False
    if a.export_options is b.export_options:
        return True
    if a.export_options is not None and b.export_options is not None:
        return a.export_options.dynamic_shapes == b.export_options.dynamic_shapes and a.export_options.op_level_debug == b.export_options.op_level_debug and (a.export_options.diagnostic_options == b.export_options.diagnostic_options) and (a.export_options.onnx_registry is b.export_options.onnx_registry) and (a.export_options.fake_context is b.export_options.fake_context)
    return False