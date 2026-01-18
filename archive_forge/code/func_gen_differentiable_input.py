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
def gen_differentiable_input(arg: Union[Argument, SelfArgument, TensorOptionsArguments]) -> Optional[DifferentiableInput]:
    if isinstance(arg, TensorOptionsArguments):
        return None
    a: Argument = arg.argument if isinstance(arg, SelfArgument) else arg
    cpp_type = cpp.argument_type(a, binds=a.name, symint=True).cpp_type()
    if not is_differentiable(a.name, a.type, info):
        return None
    return DifferentiableInput(name=a.name, type=a.type, cpp_type=cpp_type)