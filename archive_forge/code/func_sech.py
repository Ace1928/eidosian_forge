from __future__ import annotations
import functools
import numpy as np
from qiskit.pulse.exceptions import PulseError
def sech(times: np.ndarray, amp: complex, center: float, sigma: float, zeroed_width: float | None=None, rescale_amp: bool=False, ret_x: bool=False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Continuous unnormalized sech pulse.

    Args:
        times: Times to output pulse for.
        amp: Pulse amplitude at `center`.
        center: Center (mean) of pulse.
        sigma: Width (standard deviation) of pulse.
        zeroed_width: Subtract baseline from pulse to make sure
            $\\Omega_g(center \\pm zeroed_width/2)=0$ is satisfied. This is used to avoid
            large discontinuities at the start and end of the pulse.
        rescale_amp: If `zeroed_width` is not `None` and `rescale_amp=True` the pulse will
            be rescaled so that $\\Omega_g(center)=amp$.
        ret_x: Return centered and standard deviation normalized pulse location.
            $x=(times-center)/sigma$.
    """
    times = np.asarray(times, dtype=np.complex128)
    x = (times - center) / sigma
    sech_out = amp * sech_fn(x).astype(np.complex128)
    if zeroed_width is not None:
        sech_out = _fix_sech_width(sech_out, amp=amp, center=center, sigma=sigma, zeroed_width=zeroed_width, rescale_amp=rescale_amp)
    if ret_x:
        return (sech_out, x)
    return sech_out