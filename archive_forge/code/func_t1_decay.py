from typing import Any, Optional, TYPE_CHECKING
import warnings
import pandas as pd
import sympy
from matplotlib import pyplot as plt
import numpy as np
from cirq import circuits, ops, study, value, _import
from cirq._compat import proper_repr
def t1_decay(sampler: 'cirq.Sampler', *, qubit: 'cirq.Qid', num_points: int, max_delay: 'cirq.DURATION_LIKE', min_delay: 'cirq.DURATION_LIKE'=None, repetitions: int=1000) -> 'cirq.experiments.T1DecayResult':
    """Runs a t1 decay experiment.

    Initializes a qubit into the |1⟩ state, waits for a variable amount of time,
    and measures the qubit. Plots how often the |1⟩ state is observed for each
    amount of waiting.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        qubit: The qubit under test.
        num_points: The number of evenly spaced delays to test.
        max_delay: The largest delay to test.
        min_delay: The smallest delay to test. Defaults to no delay.
        repetitions: The number of repetitions of the circuit for each delay.

    Returns:
        A T1DecayResult object that stores and can plot the data.

    Raises:
        ValueError: If the supplied parameters are not valid: negative repetitions,
            max delay less than min, or min delay less than 0.
    """
    min_delay_dur = value.Duration(min_delay)
    max_delay_dur = value.Duration(max_delay)
    min_delay_nanos = min_delay_dur.total_nanos()
    max_delay_nanos = max_delay_dur.total_nanos()
    if repetitions <= 0:
        raise ValueError('repetitions <= 0')
    if isinstance(min_delay_nanos, sympy.Expr) or isinstance(max_delay_nanos, sympy.Expr):
        raise ValueError('min_delay and max_delay cannot be sympy expressions.')
    if max_delay_dur < min_delay_dur:
        raise ValueError('max_delay < min_delay')
    if min_delay_dur < 0:
        raise ValueError('min_delay < 0')
    var = sympy.Symbol('delay_ns')
    sweep = study.Linspace(var, start=min_delay_nanos, stop=max_delay_nanos, length=num_points)
    circuit = circuits.Circuit(ops.X(qubit), ops.wait(qubit, nanos=var), ops.measure(qubit, key='output'))
    results = sampler.sample(circuit, params=sweep, repetitions=repetitions)
    tab = pd.crosstab(results.delay_ns, results.output)
    tab.rename_axis(None, axis='columns', inplace=True)
    tab = tab.rename(columns={0: 'false_count', 1: 'true_count'}).reset_index()
    for col_index, name in [(1, 'false_count'), (2, 'true_count')]:
        if name not in tab:
            tab.insert(col_index, name, [0] * tab.shape[0])
    return T1DecayResult(tab)