from typing import Any, Callable, cast, Dict, Optional, Union
import numpy as np
import sympy
from cirq import ops
def controlled_unwrap(op: ops.ControlledOperation) -> Optional[QuirkOp]:
    sub = known_quirk_op_for_operation(op.sub_operation)
    if sub is None:
        return None
    return sub.controlled(len(op.controls))