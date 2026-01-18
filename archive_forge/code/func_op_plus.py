from __future__ import annotations
import typing as T
from ...interpreterbase import (
from ...mparser import PlusAssignmentNode
@typed_operator(MesonOperator.PLUS, object)
def op_plus(self, other: TYPE_var) -> T.List[TYPE_var]:
    if not isinstance(other, list):
        if not isinstance(self.current_node, PlusAssignmentNode):
            FeatureNew.single_use('list.<plus>', '0.60.0', self.subproject, 'The right hand operand was not a list.', location=self.current_node)
        other = [other]
    return self.held_object + other