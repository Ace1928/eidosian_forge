import re
from typing import List, Optional
import torchgen.api.python as python
from torchgen.api import cpp
from torchgen.api.types import CppSignatureGroup
from torchgen.context import with_native_function
from torchgen.gen import parse_native_yaml
from torchgen.model import NativeFunction, TensorOptionsArguments, Variant
from torchgen.utils import FileManager, mapMaybe
def maybe_optional_type(type: str, is_opt: bool) -> str:
    return f'c10::optional<{type}>' if is_opt else type