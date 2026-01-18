import sys
from typing import Union
def aggregate_mean(samples: Sequence[Number], precision: int=2) -> float:
    return round(sum(samples) / len(samples), precision)