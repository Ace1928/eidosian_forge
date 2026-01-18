import math
import operator
from collections import deque
from collections.abc import Sized
from functools import partial, reduce
from itertools import (
from random import randrange, sample, choice
from sys import hexversion
def polynomial_derivative(coefficients):
    """Compute the first derivative of a polynomial.

    Example: evaluating the derivative of x^3 - 4 * x^2 - 17 * x + 60

    >>> coefficients = [1, -4, -17, 60]
    >>> derivative_coefficients = polynomial_derivative(coefficients)
    >>> derivative_coefficients
    [3, -8, -17]
    """
    n = len(coefficients)
    powers = reversed(range(1, n))
    return list(map(operator.mul, coefficients, powers))