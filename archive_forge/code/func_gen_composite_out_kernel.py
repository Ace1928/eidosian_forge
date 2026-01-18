from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.dispatcher as dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import Binding, DispatcherSignature, Expr
from torchgen.context import with_native_function
from torchgen.model import (
from torchgen.utils import concatMap
@with_native_function
def gen_composite_out_kernel(g: NativeFunctionsGroup) -> Optional[str]:
    if 'generated' not in g.out.tags:
        return None
    sig = DispatcherSignature(g.out.func)
    target_sig = DispatcherSignature(g.functional.func)
    exprs = ', '.join([e.expr for e in translate(sig.arguments(), target_sig.arguments())])
    copy_outs = []
    out_name = 'tmp_output'
    for i, out_arg in enumerate(g.out.func.arguments.out):
        functional_return_name = out_name if len(g.functional.func.returns) == 1 else f'std::get<{i}>({out_name})'
        copy_outs.append(f'  resize_out_helper({out_arg.name}, {functional_return_name});\n  copy_arg({out_arg.name}, {functional_return_name});')
    rets = []
    for i, ret_name in enumerate(g.out.func.aliased_return_names()):
        if ret_name is not None:
            rets.append(ret_name)
        else:
            functional_return_name = out_name if len(g.functional.func.returns) == 1 else f'std::get<{i}>({out_name})'
            rets.append(functional_return_name)
    copy_outs_str = '\n'.join(copy_outs)
    return f'\n{sig.defn(name=g.out.func.name.unambiguous_name() + ('_symint' if g.out.func.has_symint() else ''))} {{\n  auto {out_name} = at::_ops::{g.functional.func.name.unambiguous_name()}::call({exprs});\n  {copy_outs_str}\n  {return_str(g.out.func.returns, rets)}\n}}\n'