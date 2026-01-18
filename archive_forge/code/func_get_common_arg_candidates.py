from collections import defaultdict
from sympy.core import Basic, Mul, Add, Pow, sympify
from sympy.core.containers import Tuple, OrderedSet
from sympy.core.exprtools import factor_terms
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import symbols, Symbol
from sympy.matrices import (MatrixBase, Matrix, ImmutableMatrix,
from sympy.matrices.expressions import (MatrixExpr, MatrixSymbol, MatMul,
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.polys.rootoftools import RootOf
from sympy.utilities.iterables import numbered_symbols, sift, \
from . import cse_opts
def get_common_arg_candidates(self, argset, min_func_i=0):
    """Return a dict whose keys are function numbers. The entries of the dict are
        the number of arguments said function has in common with
        ``argset``. Entries have at least 2 items in common.  All keys have
        value at least ``min_func_i``.
        """
    count_map = defaultdict(lambda: 0)
    if not argset:
        return count_map
    funcsets = [self.arg_to_funcset[arg] for arg in argset]
    largest_funcset = max(funcsets, key=len)
    for funcset in funcsets:
        if largest_funcset is funcset:
            continue
        for func_i in funcset:
            if func_i >= min_func_i:
                count_map[func_i] += 1
    smaller_funcs_container, larger_funcs_container = sorted([largest_funcset, count_map], key=len)
    for func_i in smaller_funcs_container:
        if count_map[func_i] < 1:
            continue
        if func_i in larger_funcs_container:
            count_map[func_i] += 1
    return {k: v for k, v in count_map.items() if v >= 2}