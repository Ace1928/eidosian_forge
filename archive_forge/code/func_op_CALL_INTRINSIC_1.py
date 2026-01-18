import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def op_CALL_INTRINSIC_1(self, state, inst):
    try:
        operand = CALL_INTRINSIC_1_Operand(inst.arg)
    except TypeError:
        raise NotImplementedError(f'op_CALL_INTRINSIC_1({inst.arg})')
    if operand == ci1op.INTRINSIC_STOPITERATION_ERROR:
        state.append(inst, operand=operand)
        state.terminate()
        return
    elif operand == ci1op.UNARY_POSITIVE:
        val = state.pop()
        res = state.make_temp()
        state.append(inst, operand=operand, value=val, res=res)
        state.push(res)
        return
    elif operand == ci1op.INTRINSIC_LIST_TO_TUPLE:
        tos = state.pop()
        res = state.make_temp()
        state.append(inst, operand=operand, const_list=tos, res=res)
        state.push(res)
        return
    else:
        raise NotImplementedError(operand)