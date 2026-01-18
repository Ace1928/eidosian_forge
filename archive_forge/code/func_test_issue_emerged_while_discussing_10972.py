import itertools
import random
from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import _af_invert
from sympy.testing.pytest import raises
from sympy.core.function import diff
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import (adjoint, conjugate, transpose)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.tensor.array import Array, ImmutableDenseNDimArray, ImmutableSparseNDimArray, MutableSparseNDimArray
from sympy.tensor.array.arrayop import tensorproduct, tensorcontraction, derive_by_array, permutedims, Flatten, \
def test_issue_emerged_while_discussing_10972():
    ua = Array([-1, 0])
    Fa = Array([[0, 1], [-1, 0]])
    po = tensorproduct(Fa, ua, Fa, ua)
    assert tensorcontraction(po, (1, 2), (4, 5)) == Array([[0, 0], [0, 1]])
    sa = symbols('a0:144')
    po = Array(sa, [2, 2, 3, 3, 2, 2])
    assert tensorcontraction(po, (0, 1), (2, 3), (4, 5)) == sa[0] + sa[108] + sa[111] + sa[124] + sa[127] + sa[140] + sa[143] + sa[16] + sa[19] + sa[3] + sa[32] + sa[35]
    assert tensorcontraction(po, (0, 1, 4, 5), (2, 3)) == sa[0] + sa[111] + sa[127] + sa[143] + sa[16] + sa[32]
    assert tensorcontraction(po, (0, 1), (4, 5)) == Array([[sa[0] + sa[108] + sa[111] + sa[3], sa[112] + sa[115] + sa[4] + sa[7], sa[11] + sa[116] + sa[119] + sa[8]], [sa[12] + sa[120] + sa[123] + sa[15], sa[124] + sa[127] + sa[16] + sa[19], sa[128] + sa[131] + sa[20] + sa[23]], [sa[132] + sa[135] + sa[24] + sa[27], sa[136] + sa[139] + sa[28] + sa[31], sa[140] + sa[143] + sa[32] + sa[35]]])
    assert tensorcontraction(po, (0, 1), (2, 3)) == Array([[sa[0] + sa[108] + sa[124] + sa[140] + sa[16] + sa[32], sa[1] + sa[109] + sa[125] + sa[141] + sa[17] + sa[33]], [sa[110] + sa[126] + sa[142] + sa[18] + sa[2] + sa[34], sa[111] + sa[127] + sa[143] + sa[19] + sa[3] + sa[35]]])