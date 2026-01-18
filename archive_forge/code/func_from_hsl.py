from __future__ import annotations
import logging # isort:skip
import colorsys
from abc import ABCMeta, abstractmethod
from math import sqrt
from re import match
from typing import TYPE_CHECKING, Union
from ..core.serialization import AnyRep, Serializable, Serializer
from ..util.deprecation import deprecated
@classmethod
def from_hsl(cls, value: HSL) -> HSL:
    """ Copy an HSL color from another HSL color value.

        Args:
            value (HSL) :
                The HSL color to copy.

        Returns:
            :class:`~bokeh.colors.hsl.HSL`

        """
    return value.copy()