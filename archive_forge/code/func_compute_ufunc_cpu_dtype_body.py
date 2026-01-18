from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.ufunc as ufunc
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.api.ufunc import UfunctorBindings
from torchgen.context import with_native_function
from torchgen.model import (
from torchgen.utils import OrderedSet
def compute_ufunc_cpu_dtype_body(g: NativeFunctionsGroup, dtype: ScalarType, inner_loops: Dict[UfuncKey, UfuncSignature], parent_ctx: Sequence[Binding]) -> str:
    assert UfuncKey.CPUScalar in inner_loops, f'{dtype}, {inner_loops.keys()}'
    assert inner_loops.keys() <= {UfuncKey.CPUScalar, UfuncKey.CPUVector}
    scalar_loop = inner_loops[UfuncKey.CPUScalar]
    vec_loop = None
    if UfuncKey.CPUVector in inner_loops:
        vec_loop = inner_loops[UfuncKey.CPUVector]
    body = []
    ctx = []
    for b in parent_ctx:
        if isinstance(b.argument, Argument) and b.argument.type != BaseType(BaseTy.Scalar):
            continue
        body.append(f'auto _s_{b.name} = {b.name}.to<scalar_t>();')
        ctx.append(Expr(f'_s_{b.name}', NamedCType(b.nctype.name, BaseCType(scalar_t))))
    if vec_loop is not None:
        for b in parent_ctx:
            if isinstance(b.argument, Argument) and b.argument.type != BaseType(BaseTy.Scalar):
                continue
            body.append(f'auto _v_{b.name} = at::vec::Vectorized<scalar_t>(_s_{b.name});')
            ctx.append(Expr(f'_v_{b.name}', NamedCType(b.nctype.name, VectorizedCType(BaseCType(scalar_t)))))
    scalar_bindings = []
    vec_bindings = []
    for a in g.functional.func.arguments.flat_non_out:
        if not a.type.is_tensor_like():
            continue
        assert a.type == BaseType(BaseTy.Tensor)
        scalar_bindings.append(Binding(name=a.name, nctype=NamedCType(a.name, BaseCType(scalar_t)), argument=a))
        if vec_loop is not None:
            vec_bindings.append(Binding(name=a.name, nctype=NamedCType(a.name, VectorizedCType(BaseCType(scalar_t))), argument=a))

    def with_ctx(b: Sequence[Binding]) -> List[Union[Expr, Binding]]:
        r: List[Union[Expr, Binding]] = []
        r.extend(ctx)
        r.extend(b)
        return r
    body_str = '\n'.join(body)
    if vec_loop is not None:
        return f'\n{body_str}\ncpu_kernel_vec(iter,\n  [=]({', '.join((b.decl() for b in scalar_bindings))}) {{ return {scalar_loop.call(with_ctx(scalar_bindings))}; }},\n  [=]({', '.join((b.decl() for b in vec_bindings))}) {{ return {vec_loop.call(with_ctx(vec_bindings))}; }}\n);\n'
    else:
        return f'\n{body_str}\ncpu_kernel(iter,\n  [=]({', '.join((b.decl() for b in scalar_bindings))}) {{ return {scalar_loop.call(with_ctx(scalar_bindings))}; }}\n);\n'