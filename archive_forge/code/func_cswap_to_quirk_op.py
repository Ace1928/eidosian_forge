from typing import Any, Callable, cast, Dict, Optional, Union
import numpy as np
import sympy
from cirq import ops
def cswap_to_quirk_op(gate: ops.CSwapGate) -> Optional[QuirkOp]:
    return QuirkOp('•', 'Swap', 'Swap', can_merge=False)