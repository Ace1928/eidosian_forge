import dataclasses
from typing import Optional, Tuple
import numpy as np
from onnx.reference.op_run import OpRun
def max_min(lhs: float, rhs: float) -> Tuple[float, float]:
    if lhs >= rhs:
        return (rhs, lhs)
    return (lhs, rhs)