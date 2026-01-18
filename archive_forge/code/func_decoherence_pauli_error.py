from typing import TYPE_CHECKING, Any, Dict, Tuple, Type, Union
import numpy as np
from cirq import ops, protocols, value
from cirq._compat import proper_repr
def decoherence_pauli_error(t1_ns: float, tphi_ns: float, gate_time_ns: float) -> float:
    """The component of Pauli error caused by decoherence on a single qubit.

    Args:
        t1_ns: T1 time in nanoseconds.
        tphi_ns: Tphi time in nanoseconds.
        gate_time_ns: Duration in nanoseconds of the gate affected by this error.

    Returns:
        Calculated Pauli error resulting from decoherence.
    """
    gamma_2 = 1 / (2 * t1_ns) + 1 / tphi_ns
    exp1 = np.exp(-gate_time_ns / t1_ns)
    exp2 = np.exp(-gate_time_ns * gamma_2)
    px = 0.25 * (1 - exp1)
    py = px
    pz = 0.5 * (1 - exp2) - px
    return px + py + pz