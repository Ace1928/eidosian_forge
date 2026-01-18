from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup
from torchgen.gen import pythonify_default
from torchgen.model import (
def cpp_dispatch_exprs(f: NativeFunction, *, python_signature: Optional[PythonSignature]=None) -> Tuple[str, ...]:
    cpp_args: Sequence[Binding] = _cpp_signature(f, method=False).arguments()
    exprs: Tuple[str, ...] = tuple()
    if not isinstance(python_signature, PythonSignatureDeprecated):
        exprs = tuple((a.name for a in cpp_args))
    else:
        exprs = tuple(filter(lambda n: n != 'out' or f.func.is_out_fn(), python_signature.deprecated_args_exprs))
    if Variant.method in f.variants:
        exprs = tuple(filter('self'.__ne__, exprs))
    return exprs