from __future__ import annotations
import dataclasses
import re
import typing
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union
import torch
from torch import _C
from torch._C import _onnx as _C_onnx
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, registration
@_beartype.beartype
def aten_op(self, operator: str, *args, overload_name: str='', **kwargs):
    """Generates an ONNX ATen op node.

        This function is for backward compatibility with the old symbolic functions.
        """
    return self.op('aten::ATen', *args, operator_s=operator, overload_name_s=overload_name, **kwargs)