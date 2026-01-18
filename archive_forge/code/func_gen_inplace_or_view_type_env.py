from typing import Dict, List, Optional, Sequence, Tuple
from torchgen.api import cpp
from torchgen.api.autograd import (
from torchgen.api.types import (
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.model import (
from torchgen.utils import FileManager
from .context import with_native_function_with_differentiability_info
from .gen_trace_type import (
def gen_inplace_or_view_type_env(fn: NativeFunctionWithDifferentiabilityInfo) -> Dict[str, List[str]]:
    definition = inplace_or_view_method_definition(fn)
    registration = inplace_or_view_method_registration(fn)
    return {'ops_headers': [f'#include <ATen/ops/{fn.func.root_name}_ops.h>'] if definition is not None else [], 'inplace_or_view_method_definitions': [definition] if definition is not None else [], 'inplace_or_view_wrapper_registrations': [registration] if registration is not None else []}