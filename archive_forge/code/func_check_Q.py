from fractions import Fraction
from decimal import Decimal
import pickle
from typing import Callable, List, Tuple, Type
from sympy.testing.pytest import raises
from sympy.external.pythonmpq import PythonMPQ
def check_Q(q):
    assert isinstance(q, TQ)
    assert isinstance(q.numerator, TZ)
    assert isinstance(q.denominator, TZ)
    return (q.numerator, q.denominator)