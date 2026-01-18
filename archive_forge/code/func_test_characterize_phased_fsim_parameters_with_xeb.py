import itertools
import multiprocessing
from typing import Iterable
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.experiments.random_quantum_circuit_generation as rqcg
from cirq.experiments.xeb_fitting import (
from cirq.experiments.xeb_sampling import sample_2q_xeb_circuits
def test_characterize_phased_fsim_parameters_with_xeb():
    q0, q1 = cirq.LineQubit.range(2)
    rs = np.random.RandomState(52)
    circuits = [rqcg.random_rotations_between_two_qubit_circuit(q0, q1, depth=20, two_qubit_op_factory=lambda a, b, _: cirq.SQRT_ISWAP(a, b), seed=rs) for _ in range(2)]
    cycle_depths = np.arange(3, 20, 6)
    sampled_df = sample_2q_xeb_circuits(sampler=cirq.Simulator(seed=rs), circuits=circuits, cycle_depths=cycle_depths, progress_bar=None)
    options = SqrtISwapXEBOptions(characterize_theta=True, characterize_gamma=False, characterize_chi=False, characterize_zeta=False, characterize_phi=False)
    p_circuits = [parameterize_circuit(circuit, options) for circuit in circuits]
    with multiprocessing.Pool() as pool:
        result = characterize_phased_fsim_parameters_with_xeb(sampled_df=sampled_df, parameterized_circuits=p_circuits, cycle_depths=cycle_depths, options=options, fatol=0.01, xatol=0.01, pool=pool)
    opt_res = result.optimization_results[q0, q1]
    assert np.abs(opt_res.x[0] + np.pi / 4) < 0.1
    assert np.abs(opt_res.fun) < 0.1
    assert len(result.fidelities_df) == len(cycle_depths)
    assert np.all(result.fidelities_df['fidelity'] > 0.95)