from .roundTools import otRound, nearestMultipleShortestRepr
import logging
def strToFixed(string, precisionBits):
    """Converts a string representing a decimal float to a fixed-point number.

    Args:
            string (str): A string representing a decimal float.
            precisionBits (int): Number of precision bits, *up to a maximum of 16*.

    Returns:
            int: Fixed-point representation.

    Examples::

            >>> ## to convert a float string to a 2.14 fixed-point number:
            >>> strToFixed('-0.61884', precisionBits=14)
            -10139
    """
    value = float(string)
    return otRound(value * (1 << precisionBits))