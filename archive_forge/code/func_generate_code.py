import collections
import textwrap
from dataclasses import dataclass, field
from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable
from xarray.core import duck_array_ops
from xarray.core.options import OPTIONS
from xarray.core.types import Dims, Self
from xarray.core.utils import contains_only_chunked_or_numpy, module_available
from __future__ import annotations
from collections.abc import Sequence
from typing import Any, Callable
from xarray.core import duck_array_ops
from xarray.core.types import Dims, Self
def generate_code(self, method, has_keep_attrs):
    extra_kwargs = [kwarg.call for kwarg in method.extra_kwargs if kwarg.call]
    if self.datastructure.numeric_only:
        extra_kwargs.append(f'numeric_only={method.numeric_only},')
    if extra_kwargs:
        extra_kwargs = textwrap.indent('\n' + '\n'.join(extra_kwargs), 12 * ' ')
    else:
        extra_kwargs = ''
    keep_attrs = '\n' + 12 * ' ' + 'keep_attrs=keep_attrs,' if has_keep_attrs else ''
    return f'        return self.reduce(\n            duck_array_ops.{method.array_method},\n            dim=dim,{extra_kwargs}{keep_attrs}\n            **kwargs,\n        )'