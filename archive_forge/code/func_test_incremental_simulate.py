import multiprocessing
from typing import Dict, Any, Optional
from typing import Sequence
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.experiments.random_quantum_circuit_generation as rqcg
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
@pytest.mark.parametrize('multiprocess', (True, False))
def test_incremental_simulate(multiprocess):
    q0, q1 = cirq.LineQubit.range(2)
    circuits = [rqcg.random_rotations_between_two_qubit_circuit(q0, q1, depth=100, two_qubit_op_factory=lambda a, b, _: cirq.SQRT_ISWAP(a, b)) for _ in range(20)]
    cycle_depths = np.arange(3, 100, 9, dtype=np.int64)
    if multiprocess:
        pool = multiprocessing.Pool()
    else:
        pool = None
    df_ref = _ref_simulate_2q_xeb_circuits(circuits=circuits, cycle_depths=cycle_depths, pool=pool)
    df = simulate_2q_xeb_circuits(circuits=circuits, cycle_depths=cycle_depths, pool=pool)
    if pool is not None:
        pool.terminate()
    pd.testing.assert_frame_equal(df_ref, df)