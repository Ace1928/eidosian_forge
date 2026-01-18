from __future__ import annotations
import logging # isort:skip
import colorsys
from abc import ABCMeta, abstractmethod
from math import sqrt
from re import match
from typing import TYPE_CHECKING, Union
from ..core.serialization import AnyRep, Serializable, Serializer
from ..util.deprecation import deprecated
def lighten(self, amount: float) -> HSL:
    """ Lighten (increase the luminance) of this color.

        Args:
            amount (float) :
                Amount to increase the luminance by (clamped above zero)

        Returns:
            :class:`~bokeh.colors.HSL`

        """
    hsl = self.copy()
    hsl.l = self.clamp(hsl.l + amount, 1)
    return self.from_hsl(hsl)