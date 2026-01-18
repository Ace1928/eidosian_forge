import functools
import logging
import math
import operator
import sympy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch._dynamo.exc import TorchDynamoException
from torch.fx.node import Argument, Target
from torch.utils._sympy.interp import sympy_interp
from torch.fx.experimental import _config as config
def new_with_shape_env(shape_env: ShapeEnv, fake) -> Any:
    if isinstance(fake, int):
        return fake
    if isinstance(fake, torch.SymInt):
        return torch.SymInt(fake.node.with_shape_env(shape_env))
    assert isinstance(fake, FakeTensorMeta)
    return FakeTensorMeta(tuple((new_with_shape_env(shape_env, s) for s in fake.size())), tuple((new_with_shape_env(shape_env, s) for s in fake.stride())), new_with_shape_env(shape_env, fake.storage_offset()), fake.is_nested)