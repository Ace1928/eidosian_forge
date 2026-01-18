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
def test_fit_exponential_decays():
    rs = np.random.RandomState(999)
    cycle_depths = np.arange(3, 100, 11)
    fidelities = 0.95 * 0.98 ** cycle_depths + rs.normal(0, 0.2)
    a, layer_fid, a_std, layer_fid_std = _fit_exponential_decay(cycle_depths, fidelities)
    np.testing.assert_allclose([a, layer_fid], [0.95, 0.98], atol=0.02)
    assert 0 < a_std < 0.2 / len(cycle_depths)
    assert 0 < layer_fid_std < 0.001