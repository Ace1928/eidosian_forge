import collections.abc
import operator
from itertools import accumulate
from sympy import Mul, Sum, Dummy, Add
from sympy.tensor.array.expressions import PermuteDims, ArrayAdd, ArrayElementwiseApplyFunc, Reshape
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, get_rank, ArrayContraction, \
from sympy.tensor.array.expressions.utils import _apply_permutation_to_list
def convert_array_to_indexed(expr, indices):
    return _ConvertArrayToIndexed().do_convert(expr, indices)