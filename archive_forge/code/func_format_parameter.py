from dataclasses import dataclass
from fractions import Fraction
from numbers import Complex
from typing import (
import numpy as np
def format_parameter(element: ParameterDesignator) -> str:
    """
    Formats a particular parameter. Essentially the same as built-in formatting except using 'i'
    instead of 'j' for the imaginary number.

    :param element: The parameter to format for Quil output.
    """
    if isinstance(element, int) or isinstance(element, np.int_):
        return repr(element)
    elif isinstance(element, float):
        return _check_for_pi(element)
    elif isinstance(element, complex):
        out = ''
        r = element.real
        i = element.imag
        if i == 0:
            return repr(r)
        if r != 0:
            out += repr(r)
        if i == 1:
            assert np.isclose(r, 0, atol=1e-14)
            out = 'i'
        elif i == -1:
            assert np.isclose(r, 0, atol=1e-14)
            out = '-i'
        elif i < 0:
            out += repr(i) + 'i'
        elif r != 0:
            out += '+' + repr(i) + 'i'
        else:
            out += repr(i) + 'i'
        return out
    elif isinstance(element, MemoryReference):
        return str(element)
    elif isinstance(element, Expression):
        return _expression_to_string(element)
    raise AssertionError('Invalid parameter: %r' % element)