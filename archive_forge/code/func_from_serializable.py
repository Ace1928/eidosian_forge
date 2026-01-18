from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, Union
from ...util.dataclasses import NotRequired, Unspecified, dataclass
from ..serialization import (
@classmethod
def from_serializable(cls, rep: dict[str, AnyRep], deserializer: Deserializer) -> Expr:
    if 'expr' not in rep:
        deserializer.error("expected 'expr' field")
    expr = deserializer.decode(rep['expr'])
    transform = deserializer.decode(rep['transform']) if 'transform' in rep else Unspecified
    units = deserializer.decode(rep['units']) if 'units' in rep else Unspecified
    return Expr(expr, transform, units)