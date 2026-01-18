from __future__ import annotations
import typing
from .bit import Bit
from .classical import expr
from .classicalregister import ClassicalRegister, Clbit
def map_target(self, target, /):
    """Map the runtime variables in a ``target`` of a :class:`.SwitchCaseOp` to the new circuit,
        as defined in the ``circuit`` argument of the initialiser of this class."""
    if isinstance(target, Clbit):
        return self.bit_map[target]
    if isinstance(target, ClassicalRegister):
        return self._map_register(target)
    return self.map_expr(target)