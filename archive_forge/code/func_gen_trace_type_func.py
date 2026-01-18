import itertools
from typing import Dict, List, Sequence, Union
from torchgen.api import cpp
from torchgen.api.types import DispatcherSignature
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.model import Argument, NativeFunction, SchemaKind, TensorOptionsArguments
from torchgen.utils import FileManager
def gen_trace_type_func(fn: NativeFunction) -> Dict[str, List[str]]:
    return {'ops_headers': [f'#include <ATen/ops/{fn.root_name}_ops.h>'], 'trace_method_definitions': [method_definition(fn)], 'trace_wrapper_registrations': [method_registration(fn)]}