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
def merge_disjoint_experiments(experiments: List[Experiment], group_merged_settings: bool=True) -> Experiment:
    """
    Merges the list of experiments into a single experiment that runs the sum of the individual
    experiment programs and contains all of the combined experiment settings.

    A group of Experiments whose programs operate on disjoint sets of qubits can be
    'parallelized' so that the total number of runs can be reduced after grouping the settings.
    Settings which act on disjoint sets of qubits can be automatically estimated from the same
    run on the quantum computer.

    If any experiment programs act on a shared qubit they cannot be thoughtlessly composed since
    the order of operations on the shared qubit may have a significant impact on the program
    behaviour; therefore we do not recommend using this method if this is the case.

    Even when the individual experiments act on disjoint sets of qubits you must be
    careful not to associate 'parallel' with 'simultaneous' execution. Physically the gates
    specified in a pyquil Program occur as soon as resources are available; meanwhile, measurement
    happens only after all gates. There is no specification of the exact timing of gates beyond
    their causal relationships. Therefore, while grouping experiments into parallel operation can
    be quite beneficial for time savings, do not depend on any simultaneous execution of gates on
    different qubits, and be wary of the fact that measurement happens only after all gates have
    finished.

    Note that to get the time saving benefits the settings must be grouped on the merged
    experiment--by default this is done before returning the experiment.

    :param experiments: a group of experiments to combine into a single experiment
    :param group_merged_settings: By default group the settings of the merged experiment.
    :return: a single experiment that runs the summed program and all settings.
    """
    used_qubits: Set[int] = set()
    for expt in experiments:
        if expt.program.get_qubits().intersection(used_qubits):
            raise ValueError('Experiment programs act on some shared set of qubits and cannot be merged unambiguously.')
        used_qubits = used_qubits.union(cast(Set[int], expt.program.get_qubits()))
    all_settings = [setting for expt in experiments for simult_settings in expt for setting in simult_settings]
    merged_program = sum([expt.program for expt in experiments], Program())
    merged_program.wrap_in_numshots_loop(max([expt.program.num_shots for expt in experiments]))
    symm_levels = [expt.symmetrization for expt in experiments]
    symm_level = max(symm_levels)
    if SymmetrizationLevel.EXHAUSTIVE in symm_levels:
        symm_level = SymmetrizationLevel.EXHAUSTIVE
    merged_expt = Experiment(all_settings, merged_program, symmetrization=symm_level)
    if group_merged_settings:
        merged_expt = group_settings(merged_expt)
    return merged_expt