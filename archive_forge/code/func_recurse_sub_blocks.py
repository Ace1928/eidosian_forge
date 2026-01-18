from collections import defaultdict
from collections.abc import Iterable
from inspect import isfunction
from functools import reduce
from sympy.assumptions.refine import refine
from sympy.core import SympifyError, Add
from sympy.core.basic import Atom
from sympy.core.decorators import call_highest_priority
from sympy.core.kind import Kind, NumberKind
from sympy.core.logic import fuzzy_and, FuzzyBool
from sympy.core.mod import Mod
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import Abs, re, im
from .utilities import _dotprodsimp, _simplify
from sympy.polys.polytools import Poly
from sympy.utilities.iterables import flatten, is_sequence
from sympy.utilities.misc import as_int, filldedent
from sympy.tensor.array import NDimArray
from .utilities import _get_intermediate_simp_bool
def recurse_sub_blocks(M):
    i = 1
    while i <= M.shape[0]:
        if i == 1:
            to_the_right = M[0, i:]
            to_the_bottom = M[i:, 0]
        else:
            to_the_right = M[:i, i:]
            to_the_bottom = M[i:, :i]
        if any(to_the_right) or any(to_the_bottom):
            i += 1
            continue
        else:
            sub_blocks.append(M[:i, :i])
            if M.shape == M[:i, :i].shape:
                return
            else:
                recurse_sub_blocks(M[i:, i:])
                return