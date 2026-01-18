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
def test_benchmark_2q_xeb_fidelities_vectorized():
    rs = np.random.RandomState(52)
    mock_records = [{'pure_probs': rs.rand(4), 'sampled_probs': rs.rand(4)} for _ in range(100)]
    df = pd.DataFrame(mock_records)

    def _summary_stats(row):
        D = 4
        row['e_u'] = np.sum(row['pure_probs'] ** 2)
        row['u_u'] = np.sum(row['pure_probs']) / D
        row['m_u'] = np.sum(row['pure_probs'] * row['sampled_probs'])
        row['y'] = row['m_u'] - row['u_u']
        row['x'] = row['e_u'] - row['u_u']
        row['numerator'] = row['x'] * row['y']
        row['denominator'] = row['x'] ** 2
        return row
    df_old = df.copy().apply(_summary_stats, axis=1)
    D = 4
    pure_probs = np.array(df['pure_probs'].to_list())
    sampled_probs = np.array(df['sampled_probs'].to_list())
    df['e_u'] = np.sum(pure_probs ** 2, axis=1)
    df['u_u'] = np.sum(pure_probs, axis=1) / D
    df['m_u'] = np.sum(pure_probs * sampled_probs, axis=1)
    df['y'] = df['m_u'] - df['u_u']
    df['x'] = df['e_u'] - df['u_u']
    df['numerator'] = df['x'] * df['y']
    df['denominator'] = df['x'] ** 2
    pd.testing.assert_frame_equal(df_old, df)