import sys
from typing import Union
def aggregate_max(samples: Sequence[Number], precision: int=2) -> Union[float, int]:
    if isinstance(samples[-1], int):
        return max(samples)
    return round(max(samples), precision)