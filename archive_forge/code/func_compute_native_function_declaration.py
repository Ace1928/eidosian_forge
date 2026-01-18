from typing import List, Optional, Union
import torchgen.api.meta as meta
import torchgen.api.structured as structured
from torchgen.api.types import kernel_signature
from torchgen.context import with_native_function_and_index
from torchgen.model import BackendIndex, NativeFunction, NativeFunctionsGroup
from torchgen.utils import mapMaybe
@with_native_function_and_index
def compute_native_function_declaration(g: Union[NativeFunctionsGroup, NativeFunction], backend_index: BackendIndex) -> List[str]:
    metadata = backend_index.get_kernel(g)
    if isinstance(g, NativeFunctionsGroup):
        if metadata is not None and metadata.structured:
            if backend_index.external:
                raise AssertionError('Structured external backend functions are not implemented yet.')
            else:
                return gen_structured(g, backend_index)
        else:
            return list(mapMaybe(lambda f: gen_unstructured(f, backend_index), g.functions()))
    else:
        x = gen_unstructured(g, backend_index)
        return [] if x is None else [x]