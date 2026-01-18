import textwrap
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from torchgen.api.translate import translate
from torchgen.api.types import DispatcherSignature
from torchgen.context import method_with_native_function
from torchgen.model import (
from torchgen.utils import mapMaybe
def gen_all_vmap_plumbing(native_functions: Sequence[NativeFunction]) -> str:
    body = '\n'.join(list(mapMaybe(ComputeBatchRulePlumbing(), native_functions)))
    return f'\n#pragma once\n#include <ATen/Operators.h>\n#include <ATen/functorch/PlumbingHelper.h>\n\nnamespace at {{ namespace functorch {{\n\n{body}\n\n}}}} // namespace at::functorch\n'