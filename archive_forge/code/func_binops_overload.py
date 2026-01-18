from __future__ import annotations
from collections.abc import Iterator, Sequence
from typing import Optional
from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Any, Callable, overload
from xarray.core import nputils, ops
from xarray.core.types import (
def binops_overload(other_type: str, overload_type: str, return_type: str='Self', type_ignore_eq: str='override') -> list[OpsType]:
    extras = {'other_type': other_type, 'return_type': return_type}
    return [([(None, None)], required_method_binary, extras), (BINOPS_NUM + BINOPS_CMP, template_binop_overload, extras | {'overload_type': overload_type, 'type_ignore': '', 'overload_type_ignore': ''}), (BINOPS_EQNE, template_binop_overload, extras | {'overload_type': overload_type, 'type_ignore': '', 'overload_type_ignore': _type_ignore(type_ignore_eq)}), ([(None, None)], unhashable, extras), (BINOPS_REFLEXIVE, template_reflexive, extras)]