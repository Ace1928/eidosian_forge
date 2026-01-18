import builtins
import copy
import dataclasses
import inspect
import io
import math
import pathlib
import sys
import typing
from enum import auto, Enum
from typing import (
import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.fx._compatibility import compatibility
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import PassManager
from torch.utils._pytree import (
from .exported_program import ExportedProgram, ModuleCallEntry, ModuleCallSignature
from .graph_signature import ExportBackwardSignature, ExportGraphSignature
def register_dataclass(cls: Type[Any]) -> None:
    """
    Registers a dataclass as a valid input/output type for :func:`torch.export.export`.

    Args:
        cls: the dataclass type to register

    Example::

        @dataclass
        class InputDataClass:
            feature: torch.Tensor
            bias: int

        class OutputDataClass:
            res: torch.Tensor

        torch.export.register_dataclass(InputDataClass)
        torch.export.register_dataclass(OutputDataClass)

        def fn(o: InputDataClass) -> torch.Tensor:
            res = res=o.feature + o.bias
            return OutputDataClass(res=res)

        ep = torch.export.export(fn, (InputDataClass(torch.ones(2, 2), 1), ))
        print(ep)

    """
    from torch._export.utils import register_dataclass_as_pytree_node
    return register_dataclass_as_pytree_node(cls)