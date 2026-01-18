import re
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.autograd import (
from torchgen.api.types import (
from torchgen.code_template import CodeTemplate
from torchgen.context import (
from torchgen.model import (
from torchgen.utils import FileManager, mapMaybe
from .context import with_native_function_with_differentiability_info_and_key
from .gen_inplace_or_view_type import (
from .gen_trace_type import (
def find_args_with_derivatives(differentiable_inputs: List[DifferentiableInput]) -> List[DifferentiableInput]:
    """Find arguments that have derivative definitions"""
    if info is None or not info.has_derivatives:
        return differentiable_inputs
    names = {name for d in info.derivatives for name in d.var_names}
    differentiable = [arg for arg in differentiable_inputs if arg.name in names]
    if len(differentiable) != len(names):
        missing = names - {arg.name for arg in differentiable}
        raise RuntimeError(f'Missing arguments for derivatives: {missing} in {info.name}')
    return differentiable