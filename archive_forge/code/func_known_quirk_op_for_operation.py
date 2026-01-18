from typing import Any, Callable, cast, Dict, Optional, Union
import numpy as np
import sympy
from cirq import ops
def known_quirk_op_for_operation(op: ops.Operation) -> Optional[QuirkOp]:
    if isinstance(op, ops.GateOperation):
        return _gate_to_quirk_op(op.gate)
    if isinstance(op, ops.ControlledOperation):
        return controlled_unwrap(op)
    return None