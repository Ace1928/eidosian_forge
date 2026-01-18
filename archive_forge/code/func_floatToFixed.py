from .roundTools import otRound, nearestMultipleShortestRepr
import logging
def floatToFixed(value, precisionBits):
    """Converts a float to a fixed-point number given the number of
    precision bits.

    Args:
            value (float): Floating point value.
            precisionBits (int): Number of precision bits.

    Returns:
            int: Fixed-point representation.

    Examples::

            >>> floatToFixed(-0.61883544921875, precisionBits=14)
            -10139
            >>> floatToFixed(-0.61884, precisionBits=14)
            -10139
    """
    return otRound(value * (1 << precisionBits))