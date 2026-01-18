from typing import Any, Callable, cast, Dict, Optional, Union
import numpy as np
import sympy
from cirq import ops
def cnot_to_quirk_op(gate: ops.CXPowGate) -> Optional[QuirkOp]:
    return x_to_quirk_op(ops.X ** gate.exponent).controlled()