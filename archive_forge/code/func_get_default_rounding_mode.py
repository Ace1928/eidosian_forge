from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def get_default_rounding_mode(ctx=None):
    """Retrieves the global default rounding mode."""
    global _dflt_rounding_mode
    if _dflt_rounding_mode == Z3_OP_FPA_RM_TOWARD_ZERO:
        return RTZ(ctx)
    elif _dflt_rounding_mode == Z3_OP_FPA_RM_TOWARD_NEGATIVE:
        return RTN(ctx)
    elif _dflt_rounding_mode == Z3_OP_FPA_RM_TOWARD_POSITIVE:
        return RTP(ctx)
    elif _dflt_rounding_mode == Z3_OP_FPA_RM_NEAREST_TIES_TO_EVEN:
        return RNE(ctx)
    elif _dflt_rounding_mode == Z3_OP_FPA_RM_NEAREST_TIES_TO_AWAY:
        return RNA(ctx)