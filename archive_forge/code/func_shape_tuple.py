from __future__ import annotations
from collections.abc import Iterable
from typing import Protocol, Union, runtime_checkable
import numpy as np
from numpy.typing import ArrayLike, NDArray
def shape_tuple(*shapes: ShapeInput) -> tuple[int, ...]:
    """
    Flatten the input into a single tuple of integers, preserving order.

    Args:
        shapes: Integers or iterables of integers, possibly nested.

    Returns:
        A tuple of integers.

    Raises:
        ValueError: If some member of ``shapes`` is not an integer or iterable.
    """
    return tuple(_flatten_to_ints(shapes))