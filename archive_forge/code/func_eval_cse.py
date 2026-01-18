from sympy.external import import_module
from sympy.testing.pytest import raises
import ctypes
import sympy
from sympy.abc import a, b, n
def eval_cse(e, sub_dict):
    tmp_dict = {}
    for tmp_name, tmp_expr in e[0]:
        e2 = tmp_expr.subs(sub_dict)
        e3 = e2.subs(tmp_dict)
        tmp_dict[tmp_name] = e3
    return [e.subs(sub_dict).subs(tmp_dict) for e in e[1]]