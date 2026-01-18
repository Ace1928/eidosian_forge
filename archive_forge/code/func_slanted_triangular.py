from typing import Iterable
import numpy
from .config import registry
@registry.schedules('slanted_triangular.v1')
def slanted_triangular(max_rate: float, num_steps: int, *, cut_frac: float=0.1, ratio: int=32, decay: float=1.0, t: float=0.0) -> Iterable[float]:
    """Yield an infinite series of values according to Howard and Ruder's
    "slanted triangular learning rate" schedule.
    """
    cut = int(num_steps * cut_frac)
    while True:
        t += 1
        if t < cut:
            p = t / cut
        else:
            p = 1 - (t - cut) / (cut * (1 / cut_frac - 1))
        learn_rate = max_rate * (1 + p * (ratio - 1)) * (1 / ratio)
        yield learn_rate