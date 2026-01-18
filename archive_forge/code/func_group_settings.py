import functools
import itertools
from operator import mul
from typing import Dict, List, Iterable, Sequence, Set, Tuple, Union, cast
import networkx as nx
from networkx.algorithms.approximation.clique import clique_removal
from pyquil.experiment._main import Experiment
from pyquil.experiment._result import ExperimentResult
from pyquil.experiment._setting import ExperimentSetting, TensorProductState, _OneQState
from pyquil.experiment._symmetrization import SymmetrizationLevel
from pyquil.paulis import PauliTerm, sI
from pyquil.quil import Program
def group_settings(experiments: Experiment, method: str='greedy') -> Experiment:
    """
    Group experiments that are diagonal in a shared tensor product basis (TPB) to minimize number
    of QPU runs.

    .. rubric:: Background

    Given some PauliTerm operator, the 'natural' tensor product basis to
    diagonalize this term is the one which diagonalizes each Pauli operator in the
    product term-by-term.

    For example, X(1) * Z(0) would be diagonal in the 'natural' tensor product basis
    ``{(|0> +/- |1>)/Sqrt[2]} * {|0>, |1>}``, whereas Z(1) * X(0) would be diagonal
    in the 'natural' tpb ``{|0>, |1>} * {(|0> +/- |1>)/Sqrt[2]}``. The two operators
    commute but are not diagonal in each others 'natural' tpb (in fact, they are
    anti-diagonal in each others 'natural' tpb). This function tests whether two
    operators given as PauliTerms are both diagonal in each others 'natural' tpb.

    Note that for the given example of X(1) * Z(0) and Z(1) * X(0), we can construct
    the following basis which simultaneously diagonalizes both operators::

      -- |0>' = |0> (|+>) + |1> (|->)
      -- |1>' = |0> (|+>) - |1> (|->)
      -- |2>' = |0> (|->) + |1> (|+>)
      -- |3>' = |0> (-|->) + |1> (|+>)

    In this basis, X Z looks like diag(1, -1, 1, -1), and Z X looks like diag(1, 1, -1, -1).
    Notice however that this basis cannot be constructed with single-qubit operations, as each
    of the basis vectors are entangled states.


    .. rubric:: Methods

    The "greedy" method will keep a running set of 'buckets' into which grouped ExperimentSettings
    will be placed. Each new ExperimentSetting considered is assigned to the first applicable
    bucket and a new bucket is created if there are no applicable buckets.

    The "clique-removal" method maps the term grouping problem onto Max Clique graph problem.
    This method constructs a NetworkX graph where an edge exists between two settings that
    share an nTPB and then uses networkx's algorithm for clique removal. This method can give
    you marginally better groupings in certain circumstances, but constructing the
    graph is pretty slow so "greedy" is the default.

    :param experiments: a tomography experiment
    :param method: method used for grouping; the allowed methods are one of
        ['greedy', 'clique-removal']
    :return: a tomography experiment with all the same settings, just grouped according to shared
        TPBs.
    """
    if method == 'greedy':
        return group_settings_greedy(experiments)
    elif method == 'clique-removal':
        return group_settings_clique_removal(experiments)
    else:
        allowed_methods = ['greedy', 'clique-removal']
        raise ValueError(f"'method' should be one of {allowed_methods}.")