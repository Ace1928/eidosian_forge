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
def add_var(self, symbol: sympy.Symbol, type: Type) -> z3.ExprRef:
    if symbol in self.symbols:
        return self.symbols[symbol]
    log.debug('new variable: %s (%s)', symbol.name, type.__name__)
    if type is int:
        var = z3.Int(symbol.name)
        if symbol.is_positive:
            self._target_exprs.add(var > 0)
    elif type is float:
        var = z3.Real(symbol.name)
    elif type is bool:
        var = z3.Bool(symbol.name)
    else:
        raise RuntimeError(f'unsupported type for Z3 variable: {type}')
    self.symbols[symbol] = var
    return var