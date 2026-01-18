import warnings
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.utils.units import apply_prefix
def duration_in_dt(duration_in_sec: float, dt_in_sec: float) -> int:
    """
    Return duration in dt.

    Args:
        duration_in_sec: duration [s] to be converted.
        dt_in_sec: duration of dt in seconds used for conversion.

    Returns:
        Duration in dt.
    """
    res = round(duration_in_sec / dt_in_sec)
    rounding_error = abs(duration_in_sec - res * dt_in_sec)
    if rounding_error > 1e-15:
        warnings.warn('Duration is rounded to %d [dt] = %e [s] from %e [s]' % (res, res * dt_in_sec, duration_in_sec), UserWarning)
    return res