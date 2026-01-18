from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.ufunc as ufunc
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.api.ufunc import UfunctorBindings
from torchgen.context import with_native_function
from torchgen.model import (
from torchgen.utils import OrderedSet
def compute_ufunc_cuda_functors(g: NativeFunctionsGroup) -> Tuple[Dict[ScalarType, Dict[UfuncKey, UfunctorSignature]], str]:
    ufunctor_sigs: Dict[ScalarType, Dict[UfuncKey, UfunctorSignature]] = {}
    ufunctors: List[str] = []
    loops = g.out.ufunc_inner_loop
    scalar_tensor_idx_lookup = {UfuncKey.CUDAFunctorOnSelf: 1, UfuncKey.CUDAFunctorOnOther: 0, UfuncKey.CUDAFunctor: None}
    if eligible_for_binary_scalar_specialization(g):
        keys = [UfuncKey.CUDAFunctorOnSelf, UfuncKey.CUDAFunctorOnOther, UfuncKey.CUDAFunctor]
    else:
        keys = [UfuncKey.CUDAFunctor]
        for k in [UfuncKey.CUDAFunctorOnSelf, UfuncKey.CUDAFunctorOnOther]:
            assert k not in loops, f'cannot use {k} on non-binary function'
    for k in keys:
        if k in loops:
            ufunctor_sig = UfunctorSignature(g, scalar_tensor_idx=scalar_tensor_idx_lookup[k], name=loops[k].name)
            for dtype in loops[k].supported_dtypes:
                ufunctor_sigs.setdefault(dtype, {})[k] = ufunctor_sig
            continue
        ufunc_name = None
        supported_dtypes: OrderedSet[ScalarType] = OrderedSet()
        for lk in [UfuncKey.ScalarOnly, UfuncKey.Generic]:
            if lk not in loops:
                continue
            if ufunc_name is None:
                ufunc_name = loops[lk].name
            else:
                assert ufunc_name == loops[lk].name, 'ScalarOnly and Generic must have same ufunc name'
            supported_dtypes |= loops[lk].supported_dtypes
        assert ufunc_name is not None
        name = f'{k}_{ufunc_name}'
        ufunctor_sig = UfunctorSignature(g, scalar_tensor_idx=scalar_tensor_idx_lookup[k], name=name)
        for dtype in supported_dtypes:
            ufunctor_sigs.setdefault(dtype, {})[k] = ufunctor_sig
        ufunc_sig = UfuncSignature(g, name=f'ufunc::{ufunc_name}', compute_t=BaseCType(opmath_t))
        apply_ctx = ufunctor_sig.fields() + ufunctor_sig.arguments().apply
        ufunctors.append(f'\ntemplate <typename scalar_t>\nstruct {ufunctor_sig.name} {{\n  using opmath_t = at::opmath_type<scalar_t>;\n  {ufunctor_sig.decl_fields()}\n  {ufunctor_sig.inline_defn_ctor()}\n  __device__ {ufunctor_sig.decl_apply()} {{\n    return {ufunc_sig.call(apply_ctx)};\n  }}\n}};\n')
    return (ufunctor_sigs, '\n'.join(ufunctors))