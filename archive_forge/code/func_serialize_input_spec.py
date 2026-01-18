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
def serialize_input_spec(self, spec: ep.InputSpec) -> InputSpec:
    if spec.kind == ep.InputKind.USER_INPUT:
        return InputSpec.create(user_input=UserInputSpec(arg=self.serialize_argument_spec(spec.arg)))
    elif spec.kind == ep.InputKind.PARAMETER:
        assert spec.target is not None
        assert isinstance(spec.arg, ep.TensorArgument)
        return InputSpec.create(parameter=InputToParameterSpec(arg=TensorArgument(name=spec.arg.name), parameter_name=spec.target))
    elif spec.kind == ep.InputKind.BUFFER:
        assert spec.target is not None
        assert isinstance(spec.arg, ep.TensorArgument)
        return InputSpec.create(buffer=InputToBufferSpec(arg=TensorArgument(name=spec.arg.name), buffer_name=spec.target))
    elif spec.kind == ep.InputKind.CONSTANT_TENSOR:
        assert spec.target is not None
        assert isinstance(spec.arg, ep.TensorArgument)
        return InputSpec.create(tensor_constant=InputToTensorConstantSpec(arg=TensorArgument(name=spec.arg.name), tensor_constant_name=spec.target))
    else:
        raise AssertionError(f'Unknown argument kind: {spec}')