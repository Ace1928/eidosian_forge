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
def get_results_by_qubit_groups(results: Iterable[ExperimentResult], qubit_groups: Sequence[Sequence[int]]) -> Dict[Tuple[int, ...], List[ExperimentResult]]:
    """
    Organizes ExperimentResults by the group of qubits on which the observable of the result acts.

    Each experiment result will be associated with a qubit group key if the observable of the
    result.setting acts on a subset of the qubits in the group. If the result does not act on a
    subset of qubits of any given group then the result is ignored.

    Note that for groups of qubits which are not pairwise disjoint, one result may be associated to
    multiple groups.

    :param qubit_groups: groups of qubits for which you want the pertinent results.
    :param results: ExperimentResults from running an Experiment
    :return: a dictionary whose keys are individual groups of qubits (as sorted tuples). The
        corresponding value is the list of experiment results whose observables measure some
        subset of that qubit group. The result order is maintained within each group.
    """
    tuple_groups = [tuple(sorted(group)) for group in qubit_groups]
    results_by_qubit_group: Dict[Tuple[int, ...], List[ExperimentResult]] = {group: [] for group in tuple_groups}
    for res in results:
        res_qs = res.setting.out_operator.get_qubits()
        for group in tuple_groups:
            if set(res_qs).issubset(set(group)):
                results_by_qubit_group[group].append(res)
    return results_by_qubit_group