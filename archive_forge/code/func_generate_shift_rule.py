import functools
import itertools
import numbers
import warnings
import numpy as np
from scipy.linalg import solve as linalg_solve
import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.ops.functions import bind_new_parameters
from pennylane.tape import QuantumScript
@functools.lru_cache()
def generate_shift_rule(frequencies, shifts=None, order=1):
    """Computes the parameter shift rule for a unitary based on its generator's eigenvalue
    frequency spectrum.

    To compute gradients of circuit parameters in variational quantum algorithms, expressions for
    cost function first derivatives with respect to the variational parameters can be cast into
    linear combinations of expectation values at shifted parameter values. The coefficients and
    shifts defining the linear combination can be obtained from the unitary generator's eigenvalue
    frequency spectrum. Details can be found in
    `Wierichs et al. (2022) <https://doi.org/10.22331/q-2022-03-30-677>`__.

    Args:
        frequencies (tuple[int or float]): The tuple of eigenvalue frequencies. Eigenvalue
            frequencies are defined as the unique positive differences obtained from a set of
            eigenvalues.
        shifts (tuple[int or float]): the tuple of shift values. If unspecified,
            equidistant shifts are assumed. If supplied, the length of this tuple should match the
            number of given frequencies.
        order (int): the order of differentiation to compute the shift rule for

    Returns:
        tuple: a tuple of coefficients and shifts describing the gradient rule for the
        parameter-shift method. For parameter :math:`\\phi`, the coefficients :math:`c_i` and the
        shifts :math:`s_i` combine to give a gradient rule of the following form:

        .. math:: \\frac{\\partial}{\\partial\\phi}f = \\sum_{i} c_i f(\\phi + s_i).

        where :math:`f(\\phi) = \\langle 0|U(\\phi)^\\dagger \\hat{O} U(\\phi)|0\\rangle`
        for some observable :math:`\\hat{O}` and the unitary :math:`U(\\phi)=e^{iH\\phi}`.

    Raises:
        ValueError: if ``frequencies`` is not a list of unique positive values, or if ``shifts``
            (if specified) is not a list of unique values the same length as ``frequencies``.

    **Examples**

    An example of obtaining the frequencies from generator eigenvalues, and obtaining the parameter
    shift rule:

    >>> eigvals = (-0.5, 0, 0, 0.5)
    >>> frequencies = eigvals_to_frequencies(eigvals)
    >>> generate_shift_rule(frequencies)
    array([[ 0.4267767 ,  1.57079633],
           [-0.4267767 , -1.57079633],
           [-0.0732233 ,  4.71238898],
           [ 0.0732233 , -4.71238898]])

    An example with explicitly specified shift values:

    >>> frequencies = (1, 2, 4)
    >>> shifts = (np.pi / 3, 2 * np.pi / 3, np.pi / 4)
    >>> generate_shift_rule(frequencies, shifts)
    array([[ 3.        ,  0.78539816],
           [-3.        , -0.78539816],
           [-2.09077028,  1.04719755],
           [ 2.09077028, -1.04719755],
           [ 0.2186308 ,  2.0943951 ],
           [-0.2186308 , -2.0943951 ]])

    Higher order shift rules (corresponding to the :math:`n`-th derivative of the parameter) can be
    requested via the ``order`` argument. For example, to extract the second order shift rule for a
    gate with generator :math:`X/2`:

    >>> eigvals = (0.5, -0.5)
    >>> frequencies = eigvals_to_frequencies(eigvals)
    >>> generate_shift_rule(frequencies, order=2)
    array([[-0.5       ,  0.        ],
           [ 0.5       , -3.14159265]])

    This corresponds to the shift rule
    :math:`\\frac{\\partial^2 f}{\\partial phi^2} = \\frac{1}{2} \\left[f(\\phi) - f(\\phi-\\pi)\\right]`.
    """
    frequencies = tuple((f for f in frequencies if f > 0))
    rule = _get_shift_rule(frequencies, shifts=shifts)
    if order > 1:
        T = frequencies_to_period(frequencies)
        rule = _iterate_shift_rule(rule, order, period=T)
    return process_shifts(rule, tol=1e-10)