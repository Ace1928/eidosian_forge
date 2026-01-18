import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_app(self, a, d, xs):
    if z3.is_int_value(a):
        return self.pp_int(a)
    elif z3.is_rational_value(a):
        return self.pp_rational(a)
    elif z3.is_algebraic_value(a):
        return self.pp_algebraic(a)
    elif z3.is_bv_value(a):
        return self.pp_bv(a)
    elif z3.is_finite_domain_value(a):
        return self.pp_fd(a)
    elif z3.is_fprm_value(a):
        return self.pp_fprm_value(a)
    elif z3.is_fp_value(a):
        return self.pp_fp_value(a)
    elif z3.is_fp(a):
        return self.pp_fp(a, d, xs)
    elif z3.is_string_value(a):
        return self.pp_string(a)
    elif z3.is_const(a):
        return self.pp_const(a)
    else:
        f = a.decl()
        k = f.kind()
        if k == Z3_OP_POWER:
            return self.pp_power(a, d, xs)
        elif k == Z3_OP_DISTINCT:
            return self.pp_distinct(a, d, xs)
        elif k == Z3_OP_SELECT:
            return self.pp_select(a, d, xs)
        elif k == Z3_OP_SIGN_EXT or k == Z3_OP_ZERO_EXT or k == Z3_OP_REPEAT:
            return self.pp_unary_param(a, d, xs)
        elif k == Z3_OP_EXTRACT:
            return self.pp_extract(a, d, xs)
        elif k == Z3_OP_RE_LOOP:
            return self.pp_loop(a, d, xs)
        elif k == Z3_OP_DT_IS:
            return self.pp_is(a, d, xs)
        elif k == Z3_OP_ARRAY_MAP:
            return self.pp_map(a, d, xs)
        elif k == Z3_OP_CONST_ARRAY:
            return self.pp_K(a, d, xs)
        elif k == Z3_OP_PB_AT_MOST:
            return self.pp_atmost(a, d, f, xs)
        elif k == Z3_OP_PB_AT_LEAST:
            return self.pp_atleast(a, d, f, xs)
        elif k == Z3_OP_PB_LE:
            return self.pp_pbcmp(a, d, f, xs)
        elif k == Z3_OP_PB_GE:
            return self.pp_pbcmp(a, d, f, xs)
        elif k == Z3_OP_PB_EQ:
            return self.pp_pbcmp(a, d, f, xs)
        elif z3.is_pattern(a):
            return self.pp_pattern(a, d, xs)
        elif self.is_infix(k):
            return self.pp_infix(a, d, xs)
        elif self.is_unary(k):
            return self.pp_unary(a, d, xs)
        else:
            return self.pp_prefix(a, d, xs)