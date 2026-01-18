import base64
import dataclasses
import io
import json
import logging
import math
import operator
import typing
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Union
import sympy
import torch
import torch.export.exported_program as ep
from torch._export.verifier import load_verifier
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental import symbolic_shapes
from torch.utils._pytree import treespec_dumps, treespec_loads
from torch.utils._sympy.value_ranges import ValueRanges
from .schema import (  # type: ignore[attr-defined]
from torch.export.exported_program import (
from .upgrade import GraphModuleOpUpgrader
def deserialize_metadata(self, metadata: Dict[str, str]) -> Dict[str, Any]:
    ret: Dict[str, Any] = {}
    if (stack_trace := metadata.get('stack_trace')):
        ret['stack_trace'] = stack_trace

    def deserialize_meta_func(serialized_target: str):
        module = None
        if serialized_target.startswith('torch.nn'):
            module = torch.nn
            serialized_target_names = serialized_target.split('.')[2:]
        elif serialized_target.startswith('torch'):
            module = torch
            serialized_target_names = serialized_target.split('.')[1:]
        else:
            return self.deserialize_operator(serialized_target)
        target = module
        for name in serialized_target_names:
            if not hasattr(target, name):
                return serialized_target
            else:
                target = getattr(target, name)
        return target
    if (nn_module_stack_str := metadata.get('nn_module_stack')):

        def import_nn_module_stack(key, path, ty):
            return (key, (path, ty))
        nn_module_stack = dict((import_nn_module_stack(*item.split(',')) for item in nn_module_stack_str.split(ST_DELIMITER)))
        ret['nn_module_stack'] = nn_module_stack
    if (source_fn_st_str := metadata.get('source_fn_stack')):
        source_fn_st = []
        for source_fn_str in source_fn_st_str.split(ST_DELIMITER):
            name, target_str = source_fn_str.split(',')
            source_fn_st.append((name, deserialize_meta_func(target_str)))
        ret['source_fn_stack'] = source_fn_st
    return ret