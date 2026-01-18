from __future__ import annotations
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression
def detach_prefix(value: float, decimal: int | None=None) -> tuple[float, str]:
    """
    Given a SI unit value, find the most suitable prefix to scale the value.

    For example, the ``value = 1.3e8`` will be converted into a tuple of ``(130.0, "M")``,
    which represents a scaled value and auxiliary unit that may be used to display the value.
    In above example, that value might be displayed as ``130 MHz`` (unit is arbitrary here).

    Example:

        >>> value, prefix = detach_prefix(1e4)
        >>> print(f"{value} {prefix}Hz")
        10 kHz

    Args:
        value: The number to find prefix.
        decimal: Optional. An arbitrary integer number to represent a precision of the value.
            If specified, it tries to round the mantissa and adjust the prefix to rounded value.
            For example, 999_999.91 will become 999.9999 k with ``decimal=4``,
            while 1.0 M with ``decimal=3`` or less.

    Returns:
        A tuple of scaled value and prefix.

    .. note::

        This may induce tiny value error due to internal representation of float object.
        See https://docs.python.org/3/tutorial/floatingpoint.html for details.

    Raises:
        ValueError: If the ``value`` is out of range.
        ValueError: If the ``value`` is not real number.
    """
    prefactors = {-15: 'f', -12: 'p', -9: 'n', -6: 'µ', -3: 'm', 0: '', 3: 'k', 6: 'M', 9: 'G', 12: 'T', 15: 'P'}
    if not np.isreal(value):
        raise ValueError(f'Input should be real number. Cannot convert {value}.')
    if np.abs(value) != 0:
        pow10 = int(np.floor(np.log10(np.abs(value)) / 3) * 3)
    else:
        pow10 = 0
    if pow10 > 0:
        mant = value / pow(10, pow10)
    else:
        mant = value * pow(10, -pow10)
    if decimal is not None:
        mant = np.round(mant, decimal)
        if mant >= 1000:
            mant /= 1000
            pow10 += 3
    if pow10 not in prefactors:
        raise ValueError(f'Value is out of range: {value}')
    return (mant, prefactors[pow10])