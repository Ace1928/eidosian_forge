from __future__ import annotations
import functools
import numpy as np
from qiskit.pulse.exceptions import PulseError
def sech_deriv(times: np.ndarray, amp: complex, center: float, sigma: float, ret_sech: bool=False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Continuous unnormalized sech derivative pulse.

    Args:
        times: Times to output pulse for.
        amp: Pulse amplitude at `center`.
        center: Center (mean) of pulse.
        sigma: Width (standard deviation) of pulse.
        ret_sech: Return sech with which derivative was taken with.
    """
    sech_out, x = sech(times, amp=amp, center=center, sigma=sigma, ret_x=True)
    sech_out_deriv = -sech_out * np.tanh(x) / sigma
    if ret_sech:
        return (sech_out_deriv, sech_out)
    return sech_out_deriv