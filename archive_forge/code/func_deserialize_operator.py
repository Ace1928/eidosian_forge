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
def deserialize_operator(self, serialized_target: str):
    if serialized_target.startswith('_operator'):
        module = operator
        serialized_target_names = serialized_target.split('.')[1:]
    elif serialized_target.startswith('torch.ops'):
        module = torch.ops
        serialized_target_names = serialized_target.split('.')[2:]
    else:
        return serialized_target
    target = module
    for name in serialized_target_names:
        if not hasattr(target, name):
            return serialized_target
        else:
            target = getattr(target, name)
    return target