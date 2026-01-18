from itertools import chain
from .coretypes import (Unit, int8, int16, int32, int64, uint8, uint16, uint32,
def maxtype(measure):
    """Get the maximum width for a particular numeric type

    Examples
    --------
    >>> maxtype(int8)
    ctype("int64")

    >>> maxtype(Option(float64))
    Option(ty=ctype("float64"))

    >>> maxtype(bool_)
    ctype("bool")

    >>> maxtype(Decimal(11, 2))
    Decimal(precision=11, scale=2)

    >>> maxtype(Option(Decimal(11, 2)))
    Option(ty=Decimal(precision=11, scale=2))

    >>> maxtype(TimeDelta(unit='ms'))
    TimeDelta(unit='ms')

    >>> maxtype(Option(TimeDelta(unit='ms')))
    Option(ty=TimeDelta(unit='ms'))
    """
    measure = measure.measure
    isoption = isinstance(measure, Option)
    if isoption:
        measure = measure.ty
    if not matches_typeset(measure, scalar) and (not isinstance(measure, (Decimal, TimeDelta))):
        raise TypeError('measure must be numeric')
    if measure == bool_:
        result = bool_
    elif isinstance(measure, (Decimal, TimeDelta)):
        result = measure
    else:
        result = max(supertype(measure).types, key=lambda x: x.itemsize)
    return Option(result) if isoption else result