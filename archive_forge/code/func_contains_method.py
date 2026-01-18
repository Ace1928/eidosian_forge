from __future__ import annotations
import typing as T
from ...interpreterbase import (
from ...mparser import PlusAssignmentNode
@noArgsFlattening
@noKwargs
@typed_pos_args('array.contains', object)
def contains_method(self, args: T.Tuple[object], kwargs: TYPE_kwargs) -> bool:

    def check_contains(el: T.List[TYPE_var]) -> bool:
        for element in el:
            if isinstance(element, list):
                found = check_contains(element)
                if found:
                    return True
            if element == args[0]:
                return True
        return False
    return check_contains(self.held_object)