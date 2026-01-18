import textwrap
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from torchgen.api.translate import translate
from torchgen.api.types import DispatcherSignature
from torchgen.context import method_with_native_function
from torchgen.model import (
from torchgen.utils import mapMaybe
def is_optional_tensor(typ: Type) -> bool:
    return isinstance(typ, OptionalType) and is_tensor(typ.elem)