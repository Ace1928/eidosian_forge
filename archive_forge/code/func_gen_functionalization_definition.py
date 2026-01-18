from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
from torchgen.api import cpp, dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import (
from torchgen.model import (
from torchgen.native_function_generation import (
from torchgen.selective_build.selector import SelectiveBuilder
def gen_functionalization_definition(selector: SelectiveBuilder, g: Union[NativeFunction, NativeFunctionsGroup, NativeFunctionsViewGroup]) -> List[str]:
    if not selector.include_all_operators:
        return []
    if isinstance(g, NativeFunctionsViewGroup):
        view_defs = []
        if not g.composite:
            assert g.view_copy is not None
            view_defs.append(emit_view_functionalization_body(g, view_inplace=False))
            if g.view_inplace is not None:
                view_defs.append(emit_view_functionalization_body(g, view_inplace=True))
        return view_defs
    elif isinstance(g, NativeFunction):
        if str(g.func.name) not in MUTABLE_OPS_NOT_USING_FUNCTIONALIZATION:
            assert g.has_composite_implicit_autograd_kernel or not modifies_arguments(g)
        return []
    else:
        mutation_defs = []
        mutation_defs.append(emit_inplace_functionalization_body(g.out, g))
        if g.inplace is not None:
            mutation_defs.append(emit_inplace_functionalization_body(g.inplace, g))
        if g.mutable is not None:
            mutation_defs.append(emit_inplace_functionalization_body(g.mutable, g))
        return mutation_defs
    return []