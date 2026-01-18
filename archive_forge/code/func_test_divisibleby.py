import operator
import typing as t
from collections import abc
from numbers import Number
from .runtime import Undefined
from .utils import pass_environment
def test_divisibleby(value: int, num: int) -> bool:
    """Check if a variable is divisible by a number."""
    return value % num == 0