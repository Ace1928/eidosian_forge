import math
from dataclasses import dataclass
from typing import (
import torch
def intervals(self) -> Iterable[Tuple[int, int]]:
    for (start, _), length in zip(super().intervals(), self.seqlen_py):
        yield (start, start + length)