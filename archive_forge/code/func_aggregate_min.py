import sys
from typing import Union
def aggregate_min(samples: Sequence[Number], precision: int=2) -> Union[float, int]:
    if isinstance(samples[-1], int):
        return min(samples)
    return round(min(samples), precision)