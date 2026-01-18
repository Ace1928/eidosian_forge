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
def test_get_initial_simplex():
    options = SqrtISwapXEBOptions()
    simplex, names = options.get_initial_simplex_and_names()
    assert names == ['theta', 'zeta', 'chi', 'gamma', 'phi']
    assert len(simplex) == len(names) + 1
    assert simplex.shape[1] == len(names)