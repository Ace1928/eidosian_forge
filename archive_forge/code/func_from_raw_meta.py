from __future__ import annotations
import abc
import collections
import copy
import operator
from typing import (
import torch
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass
from torch.utils import _pytree as pytree
@classmethod
def from_raw_meta(cls, raw_meta: Union[_FX_TRACER_NN_MODULE_META_TYPE, _DYNAMO_NN_MODULE_META_TYPE]) -> _ModuleMeta:
    if isinstance(raw_meta, tuple) and len(raw_meta) == 2 and isinstance(raw_meta[1], type):
        return _ModuleMeta.from_fx_tracer_produced_raw_meta(raw_meta)
    if isinstance(raw_meta, tuple) and len(raw_meta) == 2 and isinstance(raw_meta[1], tuple):
        return _ModuleMeta.from_dynamo_produced_raw_meta(raw_meta)
    raise TypeError(f"Unknown type of raw meta item from node.meta['nn_module_stack'].items(): {type(raw_meta)}")