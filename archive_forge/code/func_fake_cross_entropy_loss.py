import inspect
import logging
import math
import re
import types
from typing import Dict, List
from torch._streambase import _StreamBase
from ..guards import install_guard
import torch._C
import torch._refs
import torch.fx
import torch.nn
import torch.onnx.operators
from .. import config, polyfill, variables
from ..allowed_functions import torch_get_name
from ..device_interface import get_registered_device_interfaces
from ..exc import unimplemented
from ..guards import GuardBuilder
from ..utils import (
from .base import VariableTracker
from .ctx_manager import (
from .distributed import is_constant_pg_functions, is_from_local, ProcessGroupVariable
from .higher_order_ops import TorchHigherOrderOperatorVariable
from .lists import ListVariable, TupleVariable
from .torch_function import can_dispatch_torch_function, dispatch_torch_function
def fake_cross_entropy_loss(input, target):
    from .builder import wrap_fx_proxy
    return wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_function', torch.nn.functional.cross_entropy, *proxy_args_kwargs([input, target, weight, size_average, ignore_index, reduce_arg, reduction, label_smoothing], {})))