from __future__ import annotations
import functools
import numpy as np
from qiskit.pulse.exceptions import PulseError
def gaussian_deriv(times: np.ndarray, amp: complex, center: float, sigma: float, ret_gaussian: bool=False, zeroed_width: float | None=None, rescale_amp: bool=False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Continuous unnormalized gaussian derivative pulse.

    Args:
        times: Times to output pulse for.
        amp: Pulse amplitude at `center`.
        center: Center (mean) of pulse.
        sigma: Width (standard deviation) of pulse.
        ret_gaussian: Return gaussian with which derivative was taken with.
        zeroed_width: Subtract baseline of pulse to make sure
            $\\Omega_g(center \\pm zeroed_width/2)=0$ is satisfied. This is used to avoid
            large discontinuities at the start of a pulse.
        rescale_amp: If `zeroed_width` is not `None` and `rescale_amp=True` the pulse will
            be rescaled so that $\\Omega_g(center)=amp$.
    """
    gauss, x = gaussian(times, amp=amp, center=center, sigma=sigma, zeroed_width=zeroed_width, rescale_amp=rescale_amp, ret_x=True)
    gauss_deriv = -x / sigma * gauss
    if ret_gaussian:
        return (gauss_deriv, gauss)
    return gauss_deriv