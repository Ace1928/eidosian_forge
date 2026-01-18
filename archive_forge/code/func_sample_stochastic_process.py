from __future__ import annotations
from functools import singledispatch
from math import prod
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Function, Lambda)
from sympy.core.logic import fuzzy_and
from sympy.core.mul import Mul
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.functions.special.delta_functions import DiracDelta
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.logic.boolalg import (And, Or)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.tensor.indexed import Indexed
from sympy.utilities.lambdify import lambdify
from sympy.core.relational import Relational
from sympy.core.sympify import _sympify
from sympy.sets.sets import FiniteSet, ProductSet, Intersection
from sympy.solvers.solveset import solveset
from sympy.external import import_module
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable
def sample_stochastic_process(process):
    """
    This function is used to sample from stochastic process.

    Parameters
    ==========

    process: StochasticProcess
        Process used to extract the samples. It must be an instance of
        StochasticProcess

    Examples
    ========

    >>> from sympy.stats import sample_stochastic_process, DiscreteMarkovChain
    >>> from sympy import Matrix
    >>> T = Matrix([[0.5, 0.2, 0.3],[0.2, 0.5, 0.3],[0.2, 0.3, 0.5]])
    >>> Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    >>> next(sample_stochastic_process(Y)) in Y.state_space
    True
    >>> next(sample_stochastic_process(Y))  # doctest: +SKIP
    0
    >>> next(sample_stochastic_process(Y)) # doctest: +SKIP
    2

    Returns
    =======

    sample: iterator object
        iterator object containing the sample of given process

    """
    from sympy.stats.stochastic_process_types import StochasticProcess
    if not isinstance(process, StochasticProcess):
        raise ValueError('Process must be an instance of Stochastic Process')
    return process.sample()