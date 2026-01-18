from inspect import signature
import numpy as np
from scipy.optimize import brute, shgo
import pennylane as qml
@staticmethod
def min_analytic(objective_fn, freq, f0):
    """Analytically minimize a trigonometric function that depends on a
        single parameter and has a single frequency. Uses two or
        three function evaluations.

        Args:
            objective_fn (callable): Trigonometric function to minimize
            freq (float): Frequency :math:`f` in the ``objective_fn``
            f0 (float): Value of the ``objective_fn`` at zero. Reduces the
                number of calls to the function from three to two if given.

        Returns:
            float: Position of the minimum of ``objective_fn``
            float: Value of the minimum of ``objective_fn``

        The closed form expression used here was derived in
        `Vidal & Theis (2018) <https://arxiv.org/abs/1812.06323>`__ ,
        `Parrish et al (2019) <https://arxiv.org/abs/1904.03206>`__ and
        `Ostaszewski et al (2021) <https://doi.org/10.22331/q-2021-01-28-391>`__.
        We use the notation of Appendix A of the last of these references,
        although we allow for an arbitrary frequency instead of restricting
        to :math:`f=1`.
        The returned position is guaranteed to lie within :math:`(-\\pi/f, \\pi/f]`.

        The used formula for the minimization of the :math:`d-\\text{th}`
        parameter then reads

        .. math::

            \\theta^*_d &= \\underset{\\theta_d}{\\text{argmin}}\\left<H\\right>_{\\theta_d}\\\\
                  &= -\\frac{\\pi}{2f} - \\frac{1}{f}\\text{arctan2}\\left(2\\left<H\\right>_{\\theta_d=0}
                  - \\left<H\\right>_{\\theta_d=\\pi/(2f)} - \\left<H\\right>_{\\theta_d=-\\pi/(2f)},
                  \\left<H\\right>_{\\theta_d=\\pi/(2f)} - \\left<H\\right>_{\\theta_d=-\\pi/(2f)}\\right),

        """
    if f0 is None:
        f0 = objective_fn(0.0)
    shift = 0.5 * np.pi / freq
    fp = objective_fn(shift)
    fm = objective_fn(-shift)
    C = 0.5 * (fp + fm)
    B = np.arctan2(2 * f0 - fp - fm, fp - fm)
    x_min = -shift - B / freq
    A = np.sqrt((f0 - C) ** 2 + 0.25 * (fp - fm) ** 2)
    y_min = -A + C
    if x_min <= -2 * shift:
        x_min = x_min + 4 * shift
    return (x_min, y_min)