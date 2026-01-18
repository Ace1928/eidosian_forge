from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.model import (
from .types_base import Binding, CType, Expr
from torchgen.api import (
def kernel_signature(f: NativeFunction, backend_index: BackendIndex, *, prefix: str='') -> Union['NativeSignature', 'DispatcherSignature']:
    meta = backend_index.get_kernel(f)
    symint = meta is not None and meta.supports_symint()
    if symint:
        assert f.func.has_symint(), f'attempted to define symint kernel for {backend_index.dispatch_key} without SymInt in schema'
    if backend_index.external:
        return DispatcherSignature.from_schema(f.func, prefix=prefix, symint=symint)
    else:
        return NativeSignature(f.func, prefix=prefix, symint=symint)