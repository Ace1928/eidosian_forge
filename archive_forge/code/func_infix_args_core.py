import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def infix_args_core(self, a, d, xs, r):
    sz = len(r)
    k = a.decl().kind()
    p = self.get_precedence(k)
    first = True
    for child in a.children():
        child_pp = self.pp_expr(child, d + 1, xs)
        child_k = None
        if z3.is_app(child):
            child_k = child.decl().kind()
        if k == child_k and (self.is_assoc(k) or (first and self.is_left_assoc(k))):
            self.infix_args_core(child, d, xs, r)
            sz = len(r)
            if sz > self.max_args:
                return
        elif self.is_infix_unary(child_k):
            child_p = self.get_precedence(child_k)
            if p > child_p or (_is_add(k) and _is_sub(child_k)) or (_is_sub(k) and first and _is_add(child_k)):
                r.append(child_pp)
            else:
                r.append(self.add_paren(child_pp))
            sz = sz + 1
        elif z3.is_quantifier(child):
            r.append(self.add_paren(child_pp))
        else:
            r.append(child_pp)
            sz = sz + 1
        if sz > self.max_args:
            r.append(self.pp_ellipses())
            return
        first = False