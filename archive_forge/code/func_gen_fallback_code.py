import itertools
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torchgen.api.dispatcher as dispatcher
from torchgen.api.lazy import (
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import method_with_native_function
from torchgen.dest.lazy_ts_lowering import ts_lowering_body
from torchgen.model import (
def gen_fallback_code(schema: LazyIrSchema, sig: Union[DispatcherSignature, NativeSignature], overload_name: str) -> str:
    """
    Generate code that falls back to eager conditioned on a predicate
    """
    dispatcher_sig = DispatcherSignature.from_schema(schema.func)
    exprs = translate(sig.arguments(), dispatcher_sig.arguments())
    fallback_args = ',\n                '.join([a.expr for a in exprs])
    if len(overload_name):
        aten_op_str = f'ATEN_OP2({schema.aten_name}, {overload_name})'
    else:
        aten_op_str = f'ATEN_OP({schema.aten_name})'
    return f'\n        if (force_eager_fallback({aten_symbol(schema)})) {{\n            return at::native::call_fallback_fn_symint<&ltc_eager_fallback, {aten_op_str}>::call(\n                {fallback_args}\n            );\n        }}\n'