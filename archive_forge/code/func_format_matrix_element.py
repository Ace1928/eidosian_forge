import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
def format_matrix_element(element: Union[ExpressionDesignator, str]) -> str:
    """
            Formats a parameterized matrix element.

            :param element: The parameterized element to format.
            """
    if isinstance(element, (int, float, complex, np.int_)):
        return format_parameter(element)
    elif isinstance(element, str):
        return element
    elif isinstance(element, Expression):
        return str(element)
    else:
        raise TypeError('Invalid matrix element: %r' % element)