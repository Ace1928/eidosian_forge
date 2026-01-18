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
def deserialize_output_spec(self, o: OutputSpec) -> ep.OutputSpec:
    if o.user_output is not None:
        return ep.OutputSpec(kind=ep.OutputKind.USER_OUTPUT, arg=self.deserialize_argument_spec(o.user_output.arg), target=None)
    elif o.loss_output is not None:
        return ep.OutputSpec(kind=ep.OutputKind.LOSS_OUTPUT, arg=PyTensorArgument(name=o.loss_output.arg.name), target=None)
    elif o.buffer_mutation is not None:
        return ep.OutputSpec(kind=ep.OutputKind.BUFFER_MUTATION, arg=PyTensorArgument(name=o.buffer_mutation.arg.name), target=o.buffer_mutation.buffer_name)
    elif o.gradient_to_parameter is not None:
        return ep.OutputSpec(kind=ep.OutputKind.GRADIENT_TO_PARAMETER, arg=PyTensorArgument(name=o.gradient_to_parameter.arg.name), target=o.gradient_to_parameter.parameter_name)
    elif o.gradient_to_user_input is not None:
        return ep.OutputSpec(kind=ep.OutputKind.GRADIENT_TO_USER_INPUT, arg=PyTensorArgument(name=o.gradient_to_user_input.arg.name), target=o.gradient_to_user_input.user_input_name)
    else:
        raise AssertionError(f'Unknown output spec {o}')