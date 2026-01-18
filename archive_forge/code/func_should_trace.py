import itertools
from typing import Dict, List, Sequence, Union
from torchgen.api import cpp
from torchgen.api.types import DispatcherSignature
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.model import Argument, NativeFunction, SchemaKind, TensorOptionsArguments
from torchgen.utils import FileManager
def should_trace(f: NativeFunction) -> bool:
    if any((str(arg.type) in {'Storage', 'Type', 'ConstQuantizerPtr'} for arg in f.func.schema_order_arguments())):
        return False
    if not any((r.type.is_tensor_like() for r in f.func.returns)):
        return False
    return f.func.name.name.base not in DONT_RECORD_TRACE