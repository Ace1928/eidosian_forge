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
def group_settings_clique_removal(experiments: Experiment) -> Experiment:
    """
    Group experiments that are diagonal in a shared tensor product basis (TPB) to minimize number
    of QPU runs, using a graph clique removal algorithm.

    :param experiments: a tomography experiment
    :return: a tomography experiment with all the same settings, just grouped according to shared
        TPBs.
    """
    g = construct_tpb_graph(experiments)
    _, cliqs = clique_removal(g)
    new_cliqs: List[List[ExperimentSetting]] = []
    for cliq in cliqs:
        new_cliq: List[ExperimentSetting] = []
        for expt in cliq:
            new_cliq += [expt] * g.nodes[expt]['count']
        new_cliqs += [new_cliq]
    return Experiment(new_cliqs, program=experiments.program, symmetrization=experiments.symmetrization)