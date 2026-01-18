from .roundTools import otRound, nearestMultipleShortestRepr
import logging
def strToFixedToFloat(string, precisionBits):
    """Convert a string to a decimal float with fixed-point rounding.

    This first converts string to a float, then turns it into a fixed-point
    number with ``precisionBits`` fractional binary digits, then back to a
    float again.

    This is simply a shorthand for fixedToFloat(floatToFixed(float(s))).

    Args:
            string (str): A string representing a decimal float.
            precisionBits (int): Number of precision bits.

    Returns:
            float: The transformed and rounded value.

    Examples::

            >>> import math
            >>> s = '-0.61884'
            >>> bits = 14
            >>> f = strToFixedToFloat(s, precisionBits=bits)
            >>> math.isclose(f, -0.61883544921875)
            True
            >>> f == fixedToFloat(floatToFixed(float(s), precisionBits=bits), precisionBits=bits)
            True
    """
    value = float(string)
    scale = 1 << precisionBits
    return otRound(value * scale) / scale