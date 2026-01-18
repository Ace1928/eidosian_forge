from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.model import (
from .types_base import Binding, CType, Expr
from torchgen.api import (
def inner_call(self, *, reapply_views: Optional[bool]=None) -> str:
    inner_call_name = functionalization.name(self.g, is_reverse=self.is_reverse, include_namespace=True, reapply_views=reapply_views)
    arg_ctx = functionalization.outer_arguments(is_reverse=self.is_reverse)
    capture_ctx = functionalization.capture_arguments(self.g.view.func, is_reverse=self.is_reverse)
    full_ctx = arg_ctx + capture_ctx
    assert self.g.view_copy is not None
    call_bindings = functionalization.inner_arguments(self.g.view_copy.func, is_reverse=self.is_reverse)
    maybe_index = functionalization.inner_call_index(self.g.view_copy.func)
    call_exprs = [e.expr for e in translate.translate(full_ctx, call_bindings, method=False)]
    if not self.is_reverse and maybe_index is not None:
        return f'{inner_call_name}({', '.join(call_exprs)})[{maybe_index.name}];'
    else:
        return f'{inner_call_name}({', '.join(call_exprs)});'