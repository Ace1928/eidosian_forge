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
def shape_inference(self, func: NativeFunction, schema: LazyIrSchema) -> str:
    metadata = self.backend_index.get_kernel(func)
    assert metadata is not None
    all_args = schema.filtered_args()
    returns_length = len(schema.returns)
    is_view_copy_op = 'view_copy' in func.tags
    is_structured = func.structured or func.structured_delegate is not None
    if is_structured or is_view_copy_op:
        meta_out = '\nstd::vector<torch::lazy::Shape> shapes{torch::lazy::Shape(out_meta.scalar_type(), out_meta.sizes().vec())};'
        if returns_length > 1:

            def this_shape(i: int) -> str:
                return f'torch::lazy::Shape(std::get<{i}>(out_meta).scalar_type(), std::get<{i}>(out_meta).sizes().vec())'
            shapes_str = ','.join([this_shape(i) for i in range(returns_length)])
            meta_out = 'std::vector<torch::lazy::Shape> shapes{' + shapes_str + '};'
        dispatcher_sig = DispatcherSignature.from_schema(func.func)
        meta_conversion_str, meta_call_ctx = convert_to_meta_tensors(dispatcher_sig)
        meta_call_args = [e.expr for e in translate(meta_call_ctx, dispatcher_sig.arguments(), method=False)]
        if is_view_copy_op:
            assert func.has_composite_explicit_autograd_non_functional_kernel
            dispatch_ns = 'compositeexplicitautogradnonfunctional'
        else:
            dispatch_ns = 'meta'
        aten_name = schema.aten_name
        if func.func.has_symint() and metadata.supports_symint():
            aten_name += '_symint'
        shape_str = f'        {meta_conversion_str}\n        auto out_meta = at::{dispatch_ns}::{aten_name}({', '.join(meta_call_args)});\n        {meta_out}'
    else:
        shape_sig = ComputeShapeSignature(metadata.kernel, func, symint=metadata.supports_symint())
        shape_str = f'\n            auto shapes = {shape_sig.shape_call};'
    shape_str += f'\n            TORCH_INTERNAL_ASSERT(shapes.size() == {returns_length});'
    func_schema_str = 'aten::' + str(func.func)
    shape_str += f'\n            if(torch::lazy::symbolicShapeEnabled()){{\n                std::vector<torch::jit::IValue> inputs = {{ {', '.join((str(a.name) for a in all_args))} }};\n                const char* schema_str = "{func_schema_str}";\n                applySymbolicShapesOnLT(schema_str, inputs, shapes);\n            }}\n        '
    return shape_str