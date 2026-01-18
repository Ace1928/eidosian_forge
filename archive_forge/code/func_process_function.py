import re
from typing import List, Optional
import torchgen.api.python as python
from torchgen.api import cpp
from torchgen.api.types import CppSignatureGroup
from torchgen.context import with_native_function
from torchgen.gen import parse_native_yaml
from torchgen.model import NativeFunction, TensorOptionsArguments, Variant
from torchgen.utils import FileManager, mapMaybe
@with_native_function
def process_function(f: NativeFunction) -> Optional[str]:
    name = cpp.name(f.func)
    has_tensor_options = python.has_tensor_options(f)
    is_factory = has_tensor_options or name.endswith('_like')
    if Variant.function not in f.variants or not is_factory:
        return None
    cpp_sigs = CppSignatureGroup.from_native_function(f, method=False)
    sigs = [cpp_sigs.signature]
    if cpp_sigs.symint_signature is not None:
        sigs.append(cpp_sigs.symint_signature)
    r = ''
    for sig in sigs:
        formals: List[str] = []
        exprs: List[str] = []
        requires_grad = 'false'
        for arg in sig.arguments():
            qualified_type = fully_qualified_type(arg.type)
            if arg.default:
                formals.append(f'{qualified_type} {arg.name} = {arg.default}')
            else:
                formals.append(f'{qualified_type} {arg.name}')
            if isinstance(arg.argument, TensorOptionsArguments):
                exprs.append(f'at::TensorOptions({arg.name}).requires_grad(c10::nullopt)')
                requires_grad = f'{arg.name}.requires_grad()'
            else:
                exprs.append(arg.name)
        r += f'inline at::Tensor {sig.name()}({', '.join(formals)}) {{\n  at::AutoDispatchBelowADInplaceOrView guard;\n  return autograd::make_variable(at::{sig.name()}({', '.join(exprs)}), /*requires_grad=*/{requires_grad});\n}}\n'
    return r