from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup
from torchgen.gen import pythonify_default
from torchgen.model import (
def dispatch_lambda_return_str(f: NativeFunction) -> str:
    returns_without_annotation = tuple((Return(r.name, r.type, None) for r in f.func.returns))
    return_str = cpp.returns_type(returns_without_annotation, symint=True).cpp_type()
    if return_str not in SUPPORTED_RETURN_TYPES:
        raise RuntimeError(f'{f.func.name} returns unsupported type {return_str}')
    return return_str