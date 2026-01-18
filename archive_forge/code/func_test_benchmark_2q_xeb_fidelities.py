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
@pytest.mark.parametrize('pass_cycle_depths', (True, False))
def test_benchmark_2q_xeb_fidelities(circuits_cycle_depths_sampled_df, pass_cycle_depths):
    circuits, cycle_depths, sampled_df = circuits_cycle_depths_sampled_df
    if pass_cycle_depths:
        fid_df = benchmark_2q_xeb_fidelities(sampled_df, circuits, cycle_depths)
    else:
        fid_df = benchmark_2q_xeb_fidelities(sampled_df, circuits)
    assert len(fid_df) == len(cycle_depths)
    assert sorted(fid_df['cycle_depth'].unique()) == cycle_depths.tolist()
    assert np.all(fid_df['fidelity'] > 0.98)
    fit_df = fit_exponential_decays(fid_df)
    for _, row in fit_df.iterrows():
        assert list(row['cycle_depths']) == list(cycle_depths)
        assert len(row['fidelities']) == len(cycle_depths)