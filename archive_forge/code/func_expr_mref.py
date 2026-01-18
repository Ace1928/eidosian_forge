from numbers import Real
import numpy as np
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference, Expression, Sub, Div
from pyquil.quilbase import (
from rpcq.messages import ParameterSpec, ParameterAref, RewriteArithmeticResponse
from typing import Dict, Union, List, no_type_check
def expr_mref(expr: object) -> MemoryReference:
    """Get a suitable MemoryReference for a given expression."""
    nonlocal mref_idx
    expr = str(expr)
    if expr in seen_exprs:
        return seen_exprs[expr]
    new_mref = MemoryReference(mref_name, mref_idx)
    seen_exprs[expr] = new_mref
    mref_idx += 1
    recalculation_table[aref(new_mref)] = expr
    return new_mref