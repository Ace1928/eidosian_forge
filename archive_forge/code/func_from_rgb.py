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
def from_rgb(cls, value: RGB) -> HSL:
    """ Create an HSL color from an RGB color value.

        Args:
            value (:class:`~bokeh.colors.RGB`) :
                The RGB color to convert.

        Returns:
            :class:`~bokeh.colors.HSL`

        """
    return value.to_hsl()