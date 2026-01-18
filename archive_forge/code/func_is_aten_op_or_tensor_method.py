import functools
import importlib
import sys
import types
import torch
from .allowed_functions import _disallowed_function_ids, is_user_defined_allowed
from .utils import hashable
from .variables import (
def is_aten_op_or_tensor_method(obj):
    return obj in get_tensor_method() or isinstance(obj, (torch._ops.OpOverloadPacket, torch._ops.OpOverload))