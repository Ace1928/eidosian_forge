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
def identify_removable_identity_matrices(expr):
    editor = _EditArrayContraction(expr)
    flag = True
    while flag:
        flag = False
        for arg_with_ind in editor.args_with_ind:
            if isinstance(arg_with_ind.element, Identity):
                k = arg_with_ind.element.shape[0]
                if arg_with_ind.indices == [None, None]:
                    continue
                elif None in arg_with_ind.indices:
                    ind = [j for j in arg_with_ind.indices if j is not None][0]
                    counted = editor.count_args_with_index(ind)
                    if counted == 1:
                        editor.insert_after(arg_with_ind, OneArray(k))
                        editor.args_with_ind.remove(arg_with_ind)
                        flag = True
                        break
                    elif counted > 2:
                        continue
                elif arg_with_ind.indices[0] == arg_with_ind.indices[1]:
                    ind = arg_with_ind.indices[0]
                    counted = editor.count_args_with_index(ind)
                    if counted > 1:
                        editor.args_with_ind.remove(arg_with_ind)
                        flag = True
                        break
                    else:
                        pass
            elif ask(Q.diagonal(arg_with_ind.element)):
                if arg_with_ind.indices == [None, None]:
                    continue
                elif None in arg_with_ind.indices:
                    pass
                elif arg_with_ind.indices[0] == arg_with_ind.indices[1]:
                    ind = arg_with_ind.indices[0]
                    counted = editor.count_args_with_index(ind)
                    if counted == 3:
                        ind_new = editor.get_new_contraction_index()
                        other_args = [j for j in editor.args_with_ind if j != arg_with_ind]
                        other_args[1].indices = [ind_new if j == ind else j for j in other_args[1].indices]
                        arg_with_ind.indices = [ind, ind_new]
                        flag = True
                        break
    return editor.to_array_contraction()