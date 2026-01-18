from __future__ import annotations
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression
def apply_prefix(value: float | ParameterExpression, unit: str) -> float | ParameterExpression:
    """
    Given a SI unit prefix and value, apply the prefix to convert to
    standard SI unit.

    Args:
        value: The number to apply prefix to.
        unit: String prefix.

    Returns:
        Converted value.

    .. note::

        This may induce tiny value error due to internal representation of float object.
        See https://docs.python.org/3/tutorial/floatingpoint.html for details.

    Raises:
        ValueError: If the ``units`` aren't recognized.
    """
    prefactors = {'f': -15, 'p': -12, 'n': -9, 'u': -6, 'µ': -6, 'm': -3, 'k': 3, 'M': 6, 'G': 9, 'T': 12, 'P': 15}
    if not unit or len(unit) == 1:
        return value
    if unit[0] not in prefactors:
        raise ValueError(f'Could not understand unit: {unit}')
    pow10 = prefactors[unit[0]]
    if pow10 < 0:
        return value / pow(10, -pow10)
    return value * pow(10, pow10)