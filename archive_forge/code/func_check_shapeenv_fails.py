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
def check_shapeenv_fails(shape_env: ShapeEnv, tracked_fakes: Optional[List[Any]]) -> Optional[ValidationException]:
    assert tracked_fakes is not None
    try:
        shape_env.produce_guards([new_with_shape_env(shape_env, a.fake) for a in tracked_fakes], [a.source for a in tracked_fakes], constraint_inputs=[a.constraint_dims for a in tracked_fakes])
        return None
    except ValidationException as e:
        return e