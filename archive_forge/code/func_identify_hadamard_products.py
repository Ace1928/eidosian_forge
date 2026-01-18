import itertools
from collections import defaultdict
from typing import Tuple as tTuple, Union as tUnion, FrozenSet, Dict as tDict, List, Optional
from functools import singledispatch
from itertools import accumulate
from sympy import MatMul, Basic, Wild, KroneckerProduct
from sympy.assumptions.ask import (Q, ask)
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.matrices.expressions.diagonal import DiagMatrix
from sympy.matrices.expressions.hadamard import hadamard_product, HadamardPower
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.special import (Identity, ZeroMatrix, OneMatrix)
from sympy.matrices.expressions.trace import Trace
from sympy.matrices.expressions.transpose import Transpose
from sympy.combinatorics.permutations import _af_invert, Permutation
from sympy.matrices.common import MatrixCommon
from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.tensor.array.expressions.array_expressions import PermuteDims, ArrayDiagonal, \
from sympy.tensor.array.expressions.utils import _get_mapping_from_subranks
def identify_hadamard_products(expr: tUnion[ArrayContraction, ArrayDiagonal]):
    editor: _EditArrayContraction = _EditArrayContraction(expr)
    map_contr_to_args: tDict[FrozenSet, List[_ArgE]] = defaultdict(list)
    map_ind_to_inds: tDict[Optional[int], int] = defaultdict(int)
    for arg_with_ind in editor.args_with_ind:
        for ind in arg_with_ind.indices:
            map_ind_to_inds[ind] += 1
        if None in arg_with_ind.indices:
            continue
        map_contr_to_args[frozenset(arg_with_ind.indices)].append(arg_with_ind)
    k: FrozenSet[int]
    v: List[_ArgE]
    for k, v in map_contr_to_args.items():
        make_trace: bool = False
        if len(k) == 1 and next(iter(k)) >= 0 and (sum([next(iter(k)) in i for i in map_contr_to_args]) == 1):
            make_trace = True
            first_element = S.One
        elif len(k) != 2:
            continue
        if len(v) == 1:
            continue
        for ind in k:
            if map_ind_to_inds[ind] <= 2:
                continue

        def check_transpose(x):
            x = [i if i >= 0 else -1 - i for i in x]
            return x == sorted(x)
        if all([map_ind_to_inds[j] == len(v) and j >= 0 for j in k]) and all([j >= 0 for j in k]):
            make_trace = True
            first_element = v[0].element
            if not check_transpose(v[0].indices):
                first_element = first_element.T
            hadamard_factors = v[1:]
        else:
            hadamard_factors = v
        hp = hadamard_product(*[i.element if check_transpose(i.indices) else Transpose(i.element) for i in hadamard_factors])
        hp_indices = v[0].indices
        if not check_transpose(hadamard_factors[0].indices):
            hp_indices = list(reversed(hp_indices))
        if make_trace:
            hp = Trace(first_element * hp.T)._normalize()
            hp_indices = []
        editor.insert_after(v[0], _ArgE(hp, hp_indices))
        for i in v:
            editor.args_with_ind.remove(i)
    return editor.to_array_contraction()