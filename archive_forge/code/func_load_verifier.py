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
def load_verifier(dialect: str) -> Optional[Type[Verifier]]:
    if dialect == 'ATEN':
        return _VerifierMeta._registry.get(dialect)
    return _VerifierMeta._registry[dialect]