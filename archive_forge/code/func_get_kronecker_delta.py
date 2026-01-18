from __future__ import annotations
from typing import Any
from functools import reduce
from math import prod
from abc import abstractmethod, ABC
from collections import defaultdict
import operator
import itertools
from sympy.core.numbers import (Integer, Rational)
from sympy.combinatorics import Permutation
from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, \
from sympy.core import Basic, Expr, sympify, Add, Mul, S
from sympy.core.containers import Tuple, Dict
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import CantSympify, _sympify
from sympy.core.operations import AssocOp
from sympy.external.gmpy import SYMPY_INTS
from sympy.matrices import eye
from sympy.utilities.exceptions import (sympy_deprecation_warning,
from sympy.utilities.decorator import memoize_property, deprecated
from sympy.utilities.iterables import sift
@deprecated('\n        The TensorIndexType.get_kronecker_delta() method is deprecated. Use\n        the TensorIndexType.delta attribute instead.\n        ', deprecated_since_version='1.5', active_deprecations_target='deprecated-tensorindextype-methods')
def get_kronecker_delta(self):
    sym2 = TensorSymmetry(get_symmetric_group_sgs(2))
    delta = TensorHead('KD', [self] * 2, sym2)
    return delta