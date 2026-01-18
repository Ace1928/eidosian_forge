from typing import List, Union
from .charsetprober import CharSetProber
from .enums import ProbingState
def approx_32bit_chars(self) -> float:
    return max(1.0, self.position / 4.0)