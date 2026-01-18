from dataclasses import dataclass
from typing import List, Optional, Sequence, TYPE_CHECKING, Dict, Any
import numpy as np
import pandas as pd
from cirq import sim, value
def simulate_2q_xeb_circuits(circuits: Sequence['cirq.Circuit'], cycle_depths: Sequence[int], param_resolver: 'cirq.ParamResolverOrSimilarType'=None, pool: Optional['multiprocessing.pool.Pool']=None, simulator: Optional['cirq.SimulatesIntermediateState']=None):
    """Simulate two-qubit XEB circuits.

    These ideal probabilities can be benchmarked against potentially noisy
    results from `sample_2q_xeb_circuits`.

    Args:
        circuits: A library of two-qubit circuits generated from
            `random_rotations_between_two_qubit_circuit` of sufficient length for `cycle_depths`.
        cycle_depths: A sequence of cycle depths at which we will truncate each of the `circuits`
            to simulate.
        param_resolver: If circuits contain parameters, resolve according to this ParamResolver
            prior to simulation
        pool: If provided, execute the simulations in parallel.
        simulator: A noiseless simulator used to simulate the circuits. By default, this is
            `cirq.Simulator`. The simulator must support the `cirq.SimulatesIntermediateState`
            interface.

    Returns:
        A dataframe with index ['circuit_i', 'cycle_depth'] and column
        "pure_probs" containing the pure-state probabilities for each row.
    """
    if simulator is None:
        simulator = sim.Simulator(seed=np.random.RandomState())
    _simulate_2q_xeb_circuit = _Simulate_2q_XEB_Circuit(simulator=simulator)
    tasks = tuple((_Simulate2qXEBTask(circuit_i=circuit_i, cycle_depths=cycle_depths, circuit=circuit, param_resolver=param_resolver) for circuit_i, circuit in enumerate(circuits)))
    if pool is not None:
        nested_records = pool.map(_simulate_2q_xeb_circuit, tasks)
    else:
        nested_records = [_simulate_2q_xeb_circuit(task) for task in tasks]
    records = [record for sublist in nested_records for record in sublist]
    return pd.DataFrame(records).set_index(['circuit_i', 'cycle_depth'])