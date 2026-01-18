import dataclasses
from abc import abstractmethod, ABC
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, ops, protocols, _import
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
def per_cycle_depth(df):
    """This function is applied per cycle_depth in the following groupby aggregation."""
    fid_lsq = df['numerator'].sum() / df['denominator'].sum()
    ret = {'fidelity': fid_lsq}

    def _try_keep(k):
        """If all the values for a key `k` are the same in this group, we can keep it."""
        if k not in df.columns:
            return
        vals = df[k].unique()
        if len(vals) == 1:
            ret[k] = vals[0]
        else:
            raise AssertionError(f'When computing per-cycle-depth fidelity, multiple values for {k} were grouped together: {vals}')
    _try_keep('pair')
    return pd.Series(ret)