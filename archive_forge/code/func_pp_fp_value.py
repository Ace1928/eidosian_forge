import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_fp_value(self, a):
    _z3_assert(isinstance(a, z3.FPNumRef), 'type mismatch')
    if not self.fpa_pretty:
        r = []
        if a.isNaN():
            r.append(to_format(_z3_op_to_fpa_normal_str[Z3_OP_FPA_NAN]))
            r.append(to_format('('))
            r.append(to_format(a.sort()))
            r.append(to_format(')'))
            return compose(r)
        elif a.isInf():
            if a.isNegative():
                r.append(to_format(_z3_op_to_fpa_normal_str[Z3_OP_FPA_MINUS_INF]))
            else:
                r.append(to_format(_z3_op_to_fpa_normal_str[Z3_OP_FPA_PLUS_INF]))
            r.append(to_format('('))
            r.append(to_format(a.sort()))
            r.append(to_format(')'))
            return compose(r)
        elif a.isZero():
            if a.isNegative():
                return to_format('-zero')
            else:
                return to_format('+zero')
        else:
            _z3_assert(z3.is_fp_value(a), 'expecting FP num ast')
            r = []
            sgn = c_int(0)
            sgnb = Z3_fpa_get_numeral_sign(a.ctx_ref(), a.ast, byref(sgn))
            exp = Z3_fpa_get_numeral_exponent_string(a.ctx_ref(), a.ast, False)
            sig = Z3_fpa_get_numeral_significand_string(a.ctx_ref(), a.ast)
            r.append(to_format('FPVal('))
            if sgnb and sgn.value != 0:
                r.append(to_format('-'))
            r.append(to_format(sig))
            r.append(to_format('*(2**'))
            r.append(to_format(exp))
            r.append(to_format(', '))
            r.append(to_format(a.sort()))
            r.append(to_format('))'))
            return compose(r)
    elif a.isNaN():
        return to_format(_z3_op_to_fpa_pretty_str[Z3_OP_FPA_NAN])
    elif a.isInf():
        if a.isNegative():
            return to_format(_z3_op_to_fpa_pretty_str[Z3_OP_FPA_MINUS_INF])
        else:
            return to_format(_z3_op_to_fpa_pretty_str[Z3_OP_FPA_PLUS_INF])
    elif a.isZero():
        if a.isNegative():
            return to_format(_z3_op_to_fpa_pretty_str[Z3_OP_FPA_MINUS_ZERO])
        else:
            return to_format(_z3_op_to_fpa_pretty_str[Z3_OP_FPA_PLUS_ZERO])
    else:
        _z3_assert(z3.is_fp_value(a), 'expecting FP num ast')
        r = []
        sgn = ctypes.c_int(0)
        sgnb = Z3_fpa_get_numeral_sign(a.ctx_ref(), a.ast, byref(sgn))
        exp = Z3_fpa_get_numeral_exponent_string(a.ctx_ref(), a.ast, False)
        sig = Z3_fpa_get_numeral_significand_string(a.ctx_ref(), a.ast)
        if sgnb and sgn.value != 0:
            r.append(to_format('-'))
        r.append(to_format(sig))
        if exp != '0':
            r.append(to_format('*(2**'))
            r.append(to_format(exp))
            r.append(to_format(')'))
        return compose(r)