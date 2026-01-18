import copy
import torch
import torch.nn as nn
from torch.ao.quantization import (
from torch.ao.quantization.backend_config import (
from torch.ao.quantization.fake_quantize import (
from torch.ao.quantization.observer import (
from torch.ao.quantization.qconfig import (
from torch.ao.quantization.stubs import DeQuantStub
from torch.ao.quantization.utils import (
from torch.ao.quantization.observer import _is_activation_post_process
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.fx import GraphModule, map_arg
from torch.fx.graph import (
from .custom_config import PrepareCustomConfig
from ._decomposed import quantized_decomposed_lib  # noqa: F401
from typing import Callable, Optional, List, Dict, Any, Set, Tuple, Union, Type
from dataclasses import dataclass
from collections import namedtuple
import operator
import warnings
def get_qconv_prepack_op(conv_op: Callable) -> Callable:
    prepack_ops = {torch.nn.functional.conv1d: torch.ops.quantized.conv1d_prepack, torch.nn.functional.conv2d: torch.ops.quantized.conv2d_prepack, torch.nn.functional.conv3d: torch.ops.quantized.conv3d_prepack, torch.nn.functional.conv_transpose1d: torch.ops.quantized.conv_transpose1d_prepack, torch.nn.functional.conv_transpose2d: torch.ops.quantized.conv_transpose2d_prepack, torch.nn.functional.conv_transpose3d: torch.ops.quantized.conv_transpose3d_prepack}
    prepack_op = prepack_ops.get(conv_op, None)
    assert prepack_op, f"Didn't find prepack op for {conv_op}"
    return prepack_op