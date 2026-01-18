from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
from torchgen.api import cpp, dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import (
from torchgen.model import (
from torchgen.native_function_generation import (
from torchgen.selective_build.selector import SelectiveBuilder
@with_native_function
def emit_registration_helper(f: NativeFunction) -> str:
    assert not f.has_composite_implicit_autograd_kernel
    registration_str = f'TORCH_FN(functionalization::{wrapper_name(f.func)})'
    return f'm.impl("{f.func.name}", {registration_str});'