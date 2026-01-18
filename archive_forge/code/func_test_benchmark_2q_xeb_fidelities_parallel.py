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
def test_benchmark_2q_xeb_fidelities_parallel():
    circuits = rqcg.generate_library_of_2q_circuits(n_library_circuits=5, two_qubit_gate=cirq.ISWAP ** 0.5, max_cycle_depth=4)
    cycle_depths = [2, 3, 4]
    graph = _gridqubits_to_graph_device(cirq.GridQubit.rect(2, 2))
    combs = rqcg.get_random_combinations_for_device(n_library_circuits=len(circuits), n_combinations=2, device_graph=graph, random_state=10)
    sampled_df = sample_2q_xeb_circuits(sampler=cirq.Simulator(), circuits=circuits, cycle_depths=cycle_depths, combinations_by_layer=combs)
    fid_df = benchmark_2q_xeb_fidelities(sampled_df, circuits, cycle_depths)
    n_pairs = sum((len(c.pairs) for c in combs))
    assert len(fid_df) == len(cycle_depths) * n_pairs
    fit_df = fit_exponential_decays(fid_df)
    for _, row in fit_df.iterrows():
        assert list(row['cycle_depths']) == list(cycle_depths)
        assert len(row['fidelities']) == len(cycle_depths)