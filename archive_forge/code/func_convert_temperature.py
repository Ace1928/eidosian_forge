from __future__ import annotations
import math as _math
from typing import TYPE_CHECKING, Any
from ._codata import value as _cd
import numpy as _np
def convert_temperature(val: npt.ArrayLike, old_scale: str, new_scale: str) -> Any:
    """
    Convert from a temperature scale to another one among Celsius, Kelvin,
    Fahrenheit, and Rankine scales.

    Parameters
    ----------
    val : array_like
        Value(s) of the temperature(s) to be converted expressed in the
        original scale.
    old_scale : str
        Specifies as a string the original scale from which the temperature
        value(s) will be converted. Supported scales are Celsius ('Celsius',
        'celsius', 'C' or 'c'), Kelvin ('Kelvin', 'kelvin', 'K', 'k'),
        Fahrenheit ('Fahrenheit', 'fahrenheit', 'F' or 'f'), and Rankine
        ('Rankine', 'rankine', 'R', 'r').
    new_scale : str
        Specifies as a string the new scale to which the temperature
        value(s) will be converted. Supported scales are Celsius ('Celsius',
        'celsius', 'C' or 'c'), Kelvin ('Kelvin', 'kelvin', 'K', 'k'),
        Fahrenheit ('Fahrenheit', 'fahrenheit', 'F' or 'f'), and Rankine
        ('Rankine', 'rankine', 'R', 'r').

    Returns
    -------
    res : float or array of floats
        Value(s) of the converted temperature(s) expressed in the new scale.

    Notes
    -----
    .. versionadded:: 0.18.0

    Examples
    --------
    >>> from scipy.constants import convert_temperature
    >>> import numpy as np
    >>> convert_temperature(np.array([-40, 40]), 'Celsius', 'Kelvin')
    array([ 233.15,  313.15])

    """
    if old_scale.lower() in ['celsius', 'c']:
        tempo = _np.asanyarray(val) + zero_Celsius
    elif old_scale.lower() in ['kelvin', 'k']:
        tempo = _np.asanyarray(val)
    elif old_scale.lower() in ['fahrenheit', 'f']:
        tempo = (_np.asanyarray(val) - 32) * 5 / 9 + zero_Celsius
    elif old_scale.lower() in ['rankine', 'r']:
        tempo = _np.asanyarray(val) * 5 / 9
    else:
        raise NotImplementedError('%s scale is unsupported: supported scales are Celsius, Kelvin, Fahrenheit, and Rankine' % old_scale)
    if new_scale.lower() in ['celsius', 'c']:
        res = tempo - zero_Celsius
    elif new_scale.lower() in ['kelvin', 'k']:
        res = tempo
    elif new_scale.lower() in ['fahrenheit', 'f']:
        res = (tempo - zero_Celsius) * 9 / 5 + 32
    elif new_scale.lower() in ['rankine', 'r']:
        res = tempo * 9 / 5
    else:
        raise NotImplementedError("'%s' scale is unsupported: supported scales are 'Celsius', 'Kelvin', 'Fahrenheit', and 'Rankine'" % new_scale)
    return res