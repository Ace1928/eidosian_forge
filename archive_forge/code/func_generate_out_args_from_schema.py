from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.dispatcher as dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import Binding, DispatcherSignature, Expr
from torchgen.context import with_native_function
from torchgen.model import (
from torchgen.utils import concatMap
def generate_out_args_from_schema(func: FunctionSchema) -> Tuple[List[Return], List[Argument]]:
    assert not any((r.annotation is not None and r.annotation.is_write for r in func.returns))
    tensorlike_rets = [r for r in func.returns if r.type.is_tensor_like()]
    assert len(tensorlike_rets) > 0
    used_annotations = concatMap(lambda a: [] if a.annotation is None else a.annotation.alias_set, func.arguments.flat_all)
    valid_annotations = [x for x in 'abcdefghijklmnopqrstuvwxyz' if x not in used_annotations]
    all_rets_are_tensors = all((r.type == BaseType(BaseTy.Tensor) for r in func.returns))
    new_out_args: List[Argument] = []
    new_returns: List[Return] = []
    for i, r in enumerate(func.returns):
        if r.type.is_tensor_like():
            new_out = Argument(name='out' if len(func.returns) == 1 else f'out{i}', type=r.type, default=None, annotation=Annotation.parse(f'{valid_annotations[i]}!'))
            new_out_args.append(new_out)
            if all_rets_are_tensors:
                new_ret = Return(name=None, type=new_out.type, annotation=new_out.annotation)
                new_returns.append(new_ret)
        else:
            new_returns.append(r)
    return (new_returns, new_out_args)