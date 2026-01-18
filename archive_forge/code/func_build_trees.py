from math import prod
from sympy.core import sympify
from sympy.core.evalf import pure_complex
from sympy.core.sorting import ordered
from sympy.polys.domains import ZZ, QQ, ZZ_I, QQ_I, EX
from sympy.polys.domains.complexfield import ComplexField
from sympy.polys.domains.realfield import RealField
from sympy.polys.polyoptions import build_options
from sympy.polys.polyutils import parallel_dict_from_basic
from sympy.utilities import public
def build_trees(args):
    trees = []
    for a in args:
        if a.is_Rational:
            tree = ('Q', QQ.from_sympy(a))
        elif a.is_Add:
            tree = ('+', build_trees(a.args))
        elif a.is_Mul:
            tree = ('*', build_trees(a.args))
        else:
            tree = ('e', a)
            exts.add(a)
        trees.append(tree)
    return trees