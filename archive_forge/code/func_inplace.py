from __future__ import annotations
from collections.abc import Iterator, Sequence
from typing import Optional
from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Any, Callable, overload
from xarray.core import nputils, ops
from xarray.core.types import (
def inplace(other_type: str, type_ignore: str='') -> list[OpsType]:
    extras = {'other_type': other_type}
    return [([(None, None)], required_method_inplace, extras), (BINOPS_INPLACE, template_inplace, extras | {'type_ignore': _type_ignore(type_ignore)})]