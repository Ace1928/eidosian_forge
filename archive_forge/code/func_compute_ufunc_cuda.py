from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.ufunc as ufunc
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.api.ufunc import UfunctorBindings
from torchgen.context import with_native_function
from torchgen.model import (
from torchgen.utils import OrderedSet
@with_native_function
def compute_ufunc_cuda(g: NativeFunctionsGroup) -> str:
    ufunctor_sigs, ufunctors = compute_ufunc_cuda_functors(g)
    sig = StructuredImplSignature(g, ufunc.kernel_name(g, DispatchKey.CUDA))
    dtype_cases = []
    for dtype, inner_ufunc_sigs in ufunctor_sigs.items():
        dtype_cases.append(f'\nAT_DISPATCH_CASE(at::ScalarType::{dtype},\n  [&]() {{\n    {compute_ufunc_cuda_dtype_body(g, dtype, inner_ufunc_sigs, sig.arguments())}\n  }}\n)\n')
    dtype_cases_str = '\n'.join(dtype_cases)
    stub_sig = StubSignature(g)
    return f'\n{ufunctors}\n\n{stub_sig.type_defn()};\n{stub_sig.dispatch_decl()};\n\n{stub_sig.kernel_defn()} {{\n  AT_DISPATCH_SWITCH(iter.common_dtype(), "{sig.name}",\n    {dtype_cases_str}\n  );\n}}\nREGISTER_DISPATCH({stub_sig.name}, &{stub_sig.kernel_name});\n\n{sig.defn()} {{\n  {stub_sig.direct_call(sig.arguments())};\n}}\n'