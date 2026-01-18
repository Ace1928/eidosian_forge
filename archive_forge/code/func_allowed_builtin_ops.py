import inspect
import math
import operator
from collections.abc import Iterable
from typing import Any, Dict, final, List, Optional, Tuple, Type
import torch
from torch._ops import HigherOrderOperator, OpOverload
from torch._subclasses.fake_tensor import FakeTensor
from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import (
from torch.fx import GraphModule
from torch.fx.experimental.symbolic_shapes import SymBool, SymFloat, SymInt
def allowed_builtin_ops(self) -> List:
    return [operator.getitem, operator.add, operator.mul, operator.sub, operator.truediv, operator.ge, operator.le, operator.gt, operator.lt, operator.eq, operator.ne, operator.floordiv, operator.mod, operator.and_, operator.or_, operator.not_, operator.pow, operator.neg, operator.abs, math.ceil, math.floor]