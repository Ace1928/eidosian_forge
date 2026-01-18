import collections.abc
import operator
from collections import defaultdict, Counter
from functools import reduce
import itertools
from itertools import accumulate
from typing import Optional, List, Tuple as tTuple
import typing
from sympy.core.numbers import Integer
from sympy.core.relational import Equality
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Function, Lambda)
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import (Dummy, Symbol)
from sympy.matrices.common import MatrixCommon
from sympy.matrices.expressions.diagonal import diagonalize_vector
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.tensor.array.arrayop import (permutedims, tensorcontraction, tensordiagonal, tensorproduct)
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
from sympy.tensor.array.ndim_array import NDimArray
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.tensor.array.expressions.utils import _apply_recursively_over_nested_lists, _sort_contraction_indices, \
from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import _af_invert
from sympy.core.sympify import _sympify
def flatten_contraction_of_diagonal(self):
    if not isinstance(self.expr, ArrayDiagonal):
        return self
    contraction_down = self.expr._push_indices_down(self.expr.diagonal_indices, self.contraction_indices)
    new_contraction_indices = []
    diagonal_indices = self.expr.diagonal_indices[:]
    for i in contraction_down:
        contraction_group = list(i)
        for j in i:
            diagonal_with = [k for k in diagonal_indices if j in k]
            contraction_group.extend([l for k in diagonal_with for l in k])
            diagonal_indices = [k for k in diagonal_indices if k not in diagonal_with]
        new_contraction_indices.append(sorted(set(contraction_group)))
    new_contraction_indices = ArrayDiagonal._push_indices_up(diagonal_indices, new_contraction_indices)
    return _array_contraction(_array_diagonal(self.expr.expr, *diagonal_indices), *new_contraction_indices)