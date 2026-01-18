from typing import Iterable
import numpy
from .config import registry
@registry.schedules('compounding.v1')
def compounding(start: float, stop: float, compound: float, *, t: float=0.0) -> Iterable[float]:
    """Yield an infinite series of compounding values. Each time the
    generator is called, a value is produced by multiplying the previous
    value by the compound rate.

    EXAMPLE:
        >>> sizes = compounding(1.0, 10.0, 1.5)
        >>> assert next(sizes) == 1.
        >>> assert next(sizes) == 1 * 1.5
        >>> assert next(sizes) == 1.5 * 1.5
    """
    curr = float(start)
    while True:
        yield _clip(curr, start, stop)
        curr *= compound