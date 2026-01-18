import dataclasses
from abc import abstractmethod, ABC
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, ops, protocols, _import
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
def fit_exponential_decays(fidelities_df: pd.DataFrame) -> pd.DataFrame:
    """Fit exponential decay curves to a fidelities DataFrame.

    Args:
         fidelities_df: A DataFrame that is the result of `benchmark_2q_xeb_fidelities`. It
            may contain results for multiple pairs of qubits identified by the "pair" column.
            Each pair will be fit separately. At minimum, this dataframe must contain
            "cycle_depth", "fidelity", and "pair" columns.

    Returns:
        A new, aggregated dataframe with index given by (pair, layer_i, pair_i); columns
        for the fit parameters "a" and "layer_fid"; and nested "cycles_depths" and "fidelities"
        lists (now grouped by pair).
    """

    def _per_pair(f1):
        a, layer_fid, a_std, layer_fid_std = _fit_exponential_decay(f1['cycle_depth'], f1['fidelity'])
        record = {'a': a, 'layer_fid': layer_fid, 'cycle_depths': f1['cycle_depth'].values, 'fidelities': f1['fidelity'].values, 'a_std': a_std, 'layer_fid_std': layer_fid_std}
        return pd.Series(record)
    if 'layer_i' in fidelities_df.columns:
        groupby = ['layer_i', 'pair_i', 'pair']
    else:
        groupby = ['pair']
    return fidelities_df.groupby(groupby).apply(_per_pair)