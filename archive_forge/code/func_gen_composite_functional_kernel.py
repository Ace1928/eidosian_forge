from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.dispatcher as dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import Binding, DispatcherSignature, Expr
from torchgen.context import with_native_function
from torchgen.model import (
from torchgen.utils import concatMap
@with_native_function
def gen_composite_functional_kernel(g: NativeFunctionsGroup) -> Optional[str]:
    if 'generated' not in g.functional.tags:
        return None
    if g.inplace is not None and 'generated' not in g.inplace.tags:
        target_f = g.inplace
    elif g.mutable is not None and 'generated' not in g.mutable.tags:
        target_f = g.mutable
    else:
        raise AssertionError(str(g.functional.func))
    sig = DispatcherSignature(g.functional.func)
    target_sig = DispatcherSignature(target_f.func)
    context: List[Union[Binding, Expr]] = []
    clone_mutable_inputs = []
    cloned_return_names = []
    for a_curr, a_tgt in zip(dispatcher.jit_arguments(g.functional.func), dispatcher.jit_arguments(target_f.func)):
        if a_tgt.annotation is not None and a_tgt.annotation.is_write:
            clone_mutable_inputs.append(f'auto {a_curr.name}_clone = clone_arg({a_curr.name});')
            context.append(Expr(expr=f'{a_curr.name}_clone', type=dispatcher.argument_type(a_curr, binds=a_curr.name)))
            cloned_return_names.append(f'{a_curr.name}_clone')
        else:
            context.append(dispatcher.argument(a_curr))
    exprs = ', '.join([e.expr for e in translate(context, target_sig.arguments())])
    out_name = 'output'
    maybe_assign = f'auto {out_name} = ' if len(target_f.func.returns) > 0 else ''
    inner_return_names = gather_nonaliased_inner_rets(target_f.func, out_name)
    ret_str = return_str(g.functional.func.returns, inner_return_names + cloned_return_names)
    clone_mutable_inputs_str = '\n'.join(clone_mutable_inputs)
    return f'\n{sig.defn(name=sig.name() + ('_symint' if g.out.func.has_symint() else ''))} {{\n  {clone_mutable_inputs_str}\n  {maybe_assign}at::_ops::{target_f.func.name.unambiguous_name()}::call({exprs});\n  {ret_str}\n}}\n'