import re
from typing import List, Optional
import torchgen.api.python as python
from torchgen.api import cpp
from torchgen.api.types import CppSignatureGroup
from torchgen.context import with_native_function
from torchgen.gen import parse_native_yaml
from torchgen.model import NativeFunction, TensorOptionsArguments, Variant
from torchgen.utils import FileManager, mapMaybe
@with_native_function
def is_factory_function(f: NativeFunction) -> bool:
    if Variant.function not in f.variants:
        return False
    name = cpp.name(f.func)
    has_tensor_options = python.has_tensor_options(f)
    return has_tensor_options or name.endswith('_like')