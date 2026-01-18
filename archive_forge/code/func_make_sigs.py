from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.model import (
from .types_base import Binding, CType, Expr
from torchgen.api import (
def make_sigs(*, symint: bool) -> Tuple[CppSignature, Optional[CppSignature]]:
    faithful_signature: Optional[CppSignature] = None
    if func.arguments.tensor_options is not None or len(func.arguments.out) > 0:
        faithful_signature = make_sig(faithful=True, symint=symint)
    signature = make_sig(faithful=False, symint=symint)
    return (signature, faithful_signature)