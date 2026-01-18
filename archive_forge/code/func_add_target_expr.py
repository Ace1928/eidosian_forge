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
def add_target_expr(self, e: sympy.Expr) -> None:
    self._check_freesymbols(e)
    z3expr = self.to_z3_boolean_expr(e)
    if e not in self._target_exprs:
        log.debug('add target guard: %s', z3str(z3expr))
    self._target_exprs.add(z3expr)