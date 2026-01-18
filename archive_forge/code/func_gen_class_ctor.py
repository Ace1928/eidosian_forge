import itertools
import textwrap
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union
import torchgen.api.cpp as cpp
import torchgen.api.meta as meta
import torchgen.api.structured as structured
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import method_with_native_function, native_function_manager
from torchgen.model import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import assert_never, mapMaybe, Target
def gen_class_ctor(self, k: SchemaKind, class_name: str, returns: int) -> str:
    if k is SchemaKind.functional:
        return ''
    elif k is SchemaKind.inplace:
        return f'{class_name}(Tensor& self) : outputs_{{std::ref(self)}} {{}}'
    elif k is SchemaKind.out:
        out_args = ', '.join((f'Tensor& out{i}' for i in range(returns)))
        out_refs = ', '.join((f'std::ref(out{i})' for i in range(returns)))
        return f'{class_name}({out_args}) : outputs_{{ {out_refs} }} {{}}'
    elif k is SchemaKind.mutable or k is SchemaKind.scratch:
        raise AssertionError(f'{k} structured operators are currently not supported')
    else:
        assert_never(k)