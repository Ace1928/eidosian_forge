from __future__ import annotations
import colorsys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol
from warnings import warn
import numpy as np
from ._colors import (
from .bounds import rescale
from .utils import identity
def identity_pal() -> Callable[[], Any]:
    """
    Create palette that maps values onto themselves

    Returns
    -------
    out : function
        Palette function that takes a value or sequence of values
        and returns the same values.

    Examples
    --------
    >>> palette = identity_pal()
    >>> palette(9)
    9
    >>> palette([2, 4, 6])
    [2, 4, 6]
    """
    return identity