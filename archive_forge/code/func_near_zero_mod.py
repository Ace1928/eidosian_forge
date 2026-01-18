from typing import Union, Iterable, TYPE_CHECKING
import numpy as np
def near_zero_mod(a: float, period: float, *, atol: float=1e-08) -> bool:
    half_period = period / 2
    return near_zero((a + half_period) % period - half_period, atol=atol)