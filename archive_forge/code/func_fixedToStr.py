from .roundTools import otRound, nearestMultipleShortestRepr
import logging
def fixedToStr(value, precisionBits):
    """Converts a fixed-point number to a string representing a decimal float.

    This chooses the float that has the shortest decimal representation (the least
    number of fractional decimal digits).

    For example, to convert a fixed-point number in a 2.14 format, use
    ``precisionBits=14``::

            >>> fixedToStr(-10139, precisionBits=14)
            '-0.61884'

    This is pretty slow compared to the simple division used in ``fixedToFloat``.
    Use sporadically when you need to serialize or print the fixed-point number in
    a human-readable form.
    It uses nearestMultipleShortestRepr under the hood.

    Args:
            value (int): The fixed-point value to convert.
            precisionBits (int): Number of precision bits, *up to a maximum of 16*.

    Returns:
            str: A string representation of the value.
    """
    scale = 1 << precisionBits
    return nearestMultipleShortestRepr(value / scale, factor=1.0 / scale)