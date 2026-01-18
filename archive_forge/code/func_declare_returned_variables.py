import itertools
from typing import Dict, List, Sequence, Union
from torchgen.api import cpp
from torchgen.api.types import DispatcherSignature
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.model import Argument, NativeFunction, SchemaKind, TensorOptionsArguments
from torchgen.utils import FileManager
def declare_returned_variables(f: NativeFunction) -> str:
    modifies_arguments = f.func.kind() in (SchemaKind.inplace, SchemaKind.out)
    if modifies_arguments:
        return ''
    if len(f.func.returns) == 1:
        return ''
    types = [cpp.return_type(r, symint=True) for r in f.func.returns]
    names = cpp.return_names(f)
    return '\n'.join((f'{type.cpp_type()} {name};' for type, name in zip(types, names)))